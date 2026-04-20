import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn_v3_lite"
        self.device = device

        vector_dim = Config.VECTOR_FEATURE_LEN
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # 这一版先做“温和瘦身”：
        # 1) 地图分支通道数整体下调
        # 2) 去掉多层 ResidualBlock，保留 3 层卷积主干
        # 3) fusion / actor / critic 头同步缩窄
        # 这样改动不大，但参数量和计算量都会明显下降。
        self.vector_encoder = nn.Sequential(
            make_fc_layer(vector_dim, 96),
            nn.ReLU(),
            make_fc_layer(96, 64),
            nn.ReLU(),
        )

        self.map_encoder = nn.Sequential(
            nn.Conv2d(Config.MAP_CHANNEL, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 21 -> 10

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10 -> 5

            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for layer in self.map_encoder:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)

        self.map_fc = nn.Sequential(
            make_fc_layer(48, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            make_fc_layer(64 + 64, 128),
            nn.ReLU(),
            make_fc_layer(128, 128),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
            make_fc_layer(64, action_num),
        )

        self.critic_head = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
            make_fc_layer(64, value_num),
        )

        self.move_bias_head = nn.Sequential(
            make_fc_layer(64, 32),
            nn.ReLU(),
            make_fc_layer(32, 8),
        )

    def forward(self, vector_obs, map_obs, inference=False):
        vector_hidden = self.vector_encoder(vector_obs)

        map_hidden = self.map_encoder(map_obs)
        map_hidden = map_hidden.flatten(1)
        map_hidden = self.map_fc(map_hidden)

        hidden = torch.cat([vector_hidden, map_hidden], dim=1)
        hidden = self.fusion(hidden)

        logits = self.actor_head(hidden)
        move_bias = self.move_bias_head(map_hidden)
        logits[:, :8] = logits[:, :8] + move_bias

        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
