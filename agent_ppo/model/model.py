import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.zeros_(self.conv1.bias.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        nn.init.zeros_(self.conv2.bias.data)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(x + out)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn_v2"
        self.device = device

        vector_dim = Config.VECTOR_FEATURE_LEN
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.vector_encoder = nn.Sequential(
            make_fc_layer(vector_dim, 128),
            nn.ReLU(),
            make_fc_layer(128, 128),
            nn.ReLU(),
        )

        self.map_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.map_stage1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            nn.MaxPool2d(2),   # 36 -> 18
        )

        self.map_stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(2),   # 18 -> 9
        )

        self.map_stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
        )

        self.map_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.map_fc = nn.Sequential(
            make_fc_layer(128 * 3 * 3, 128),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            make_fc_layer(256, 256),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, action_num),
        )

        self.critic_head = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, value_num),
        )

        self.move_bias_head = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
            make_fc_layer(64, 8),
        )

    def forward(self, vector_obs, map_obs, inference=False):
        vector_hidden = self.vector_encoder(vector_obs)

        x = self.map_stem(map_obs)
        x = self.map_stage1(x)
        x = self.map_stage2(x)
        x = self.map_stage3(x)

        pooled = self.map_pool(x).flatten(1)
        map_hidden = self.map_fc(pooled)

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