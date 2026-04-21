import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.zeros_(self.conv1.bias.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        nn.init.zeros_(self.conv2.bias.data)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
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

        # dropout 配置
        self.vec_dropout_p = 0.10
        self.map_dropout_p = 0.10
        self.fusion_dropout_p = 0.10
        self.head_dropout_p = 0.10

        self.vector_encoder = nn.Sequential(
            make_fc_layer(vector_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.vec_dropout_p),
            make_fc_layer(128, 128),
            nn.ReLU(),
            nn.Dropout(self.vec_dropout_p),
        )

        self.map_stem = nn.Sequential(
            nn.Conv2d(Config.MAP_CHANNEL, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.map_dropout_p),
        )
        nn.init.orthogonal_(self.map_stem[0].weight.data)
        nn.init.zeros_(self.map_stem[0].bias.data)

        self.map_stage1 = nn.Sequential(
            ResidualBlock(32, dropout_p=self.map_dropout_p),
            nn.MaxPool2d(2),   # 21 -> 10
        )

        self.map_stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.map_dropout_p),
            ResidualBlock(64, dropout_p=self.map_dropout_p),
            nn.MaxPool2d(2),   # 10 -> 5
        )
        nn.init.orthogonal_(self.map_stage2[0].weight.data)
        nn.init.zeros_(self.map_stage2[0].bias.data)

        self.map_stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.map_dropout_p),
            ResidualBlock(128, dropout_p=self.map_dropout_p),
        )
        nn.init.orthogonal_(self.map_stage3[0].weight.data)
        nn.init.zeros_(self.map_stage3[0].bias.data)

        self.map_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.map_fc = nn.Sequential(
            make_fc_layer(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(self.map_dropout_p),
        )

        self.fusion = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            nn.Dropout(self.fusion_dropout_p),
            make_fc_layer(256, 256),
            nn.ReLU(),
            nn.Dropout(self.fusion_dropout_p),
        )

        self.actor_head = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            nn.Dropout(self.head_dropout_p),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, 64),
            nn.ReLU(),
            make_fc_layer(64, action_num),
        )

        self.critic_head = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            nn.Dropout(self.head_dropout_p),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, 32),
            nn.ReLU(),
            make_fc_layer(32, value_num),
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

        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
