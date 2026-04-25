import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_conv_layer(in_channels, out_channels, kernel_size=3, padding=1):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    nn.init.orthogonal_(conv.weight.data)
    nn.init.zeros_(conv.bias.data)
    return conv


def make_mlp(dims, dropout_p=0.0):
    layers = []
    for idx in range(len(dims) - 1):
        layers.append(make_fc_layer(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_p=0.0):
        super().__init__()
        self.conv1 = make_conv_layer(channels, channels, 3, padding=1)
        self.conv2 = make_conv_layer(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        return self.relu(x + out)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn_v3"
        self.device = device

        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        vector_layout = Config.VECTOR_FEATURES
        if vector_layout != [4, 7, 7, 8, 8, 8, 16, 4, 2]:
            raise ValueError(f"Unexpected VECTOR_FEATURES layout: {vector_layout}")

        self.vec_dropout_p = 0.10
        self.map_dropout_p = 0.10
        self.fusion_dropout_p = 0.10
        self.head_dropout_p = 0.10

        self.self_encoder = make_mlp([10, 32, 32], dropout_p=self.vec_dropout_p)
        self.monster_encoder = make_mlp([7, 32, 32], dropout_p=self.vec_dropout_p)
        self.target_encoder = make_mlp([4, 16, 16], dropout_p=self.vec_dropout_p)
        self.ray_encoder = make_mlp([8, 16, 16], dropout_p=self.vec_dropout_p)
        self.legal_encoder = make_mlp([16, 32, 32], dropout_p=self.vec_dropout_p)
        self.vector_fusion = make_mlp([208, 256, 128], dropout_p=self.fusion_dropout_p)

        self.map_stem = nn.Sequential(
            make_conv_layer(Config.MAP_CHANNEL, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.map_dropout_p),
        )

        self.map_stage1 = nn.Sequential(
            ResidualBlock(32, dropout_p=self.map_dropout_p),
            nn.MaxPool2d(2),
        )

        self.map_stage2 = nn.Sequential(
            make_conv_layer(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.map_dropout_p),
            ResidualBlock(64, dropout_p=self.map_dropout_p),
            nn.MaxPool2d(2),
        )

        self.map_stage3 = nn.Sequential(
            make_conv_layer(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.map_dropout_p),
            ResidualBlock(128, dropout_p=self.map_dropout_p),
        )

        self.map_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.map_fc = make_mlp([128 * 2 * 2, 128], dropout_p=0.0)

        self.fusion = make_mlp([256, 256, 128], dropout_p=self.fusion_dropout_p)
        self.escape_actor_head = make_mlp([128, 128, action_num], dropout_p=0.0)
        self.treasure_actor_head = make_mlp([128, 128, action_num], dropout_p=0.0)

        self.escape_critic_head = nn.Sequential(
            make_fc_layer(128, 128),
            nn.ReLU(),
            nn.Dropout(self.head_dropout_p),
            make_fc_layer(128, value_num),
        )
        self.treasure_critic_head = nn.Sequential(
            make_fc_layer(128, 128),
            nn.ReLU(),
            nn.Dropout(self.head_dropout_p),
            make_fc_layer(128, value_num),
        )

        self.escape_move_bias_head = make_mlp([128, 64, 8], dropout_p=0.0)
        self.treasure_move_bias_head = make_mlp([128, 64, 8], dropout_p=0.0)

    def _encode_vector_obs(self, vector_obs):
        hero_feat = vector_obs[:, 0:4]
        monster1_feat = vector_obs[:, 4:11]
        monster2_feat = vector_obs[:, 11:18]
        ray_feat = vector_obs[:, 18:26]
        treasure_feat = vector_obs[:, 26:34]
        buff_feat = vector_obs[:, 34:42]
        legal_feat = vector_obs[:, 42:58]
        progress_feat = vector_obs[:, 58:62]
        situation_feat = vector_obs[:, 62:64]

        shared_state_feat = torch.cat([hero_feat, progress_feat, situation_feat], dim=1)
        shared_state_hidden = self.self_encoder(shared_state_feat)

        monster1_hidden = self.monster_encoder(monster1_feat)
        monster2_hidden = self.monster_encoder(monster2_feat)
        monster_hidden = torch.cat([monster1_hidden, monster2_hidden], dim=1)

        treasure1_hidden = self.target_encoder(treasure_feat[:, 0:4])
        treasure2_hidden = self.target_encoder(treasure_feat[:, 4:8])
        buff1_hidden = self.target_encoder(buff_feat[:, 0:4])
        buff2_hidden = self.target_encoder(buff_feat[:, 4:8])
        target_hidden = torch.cat(
            [treasure1_hidden, treasure2_hidden, buff1_hidden, buff2_hidden],
            dim=1,
        )

        ray_hidden = self.ray_encoder(ray_feat)
        legal_hidden = self.legal_encoder(legal_feat)

        vector_hidden = torch.cat(
            [shared_state_hidden, monster_hidden, target_hidden, ray_hidden, legal_hidden],
            dim=1,
        )
        return self.vector_fusion(vector_hidden)

    def _apply_policy_head(self, hidden, map_hidden, actor_head, critic_head, move_bias_head):
        logits = actor_head(hidden)
        move_bias = move_bias_head(map_hidden)
        logits[:, :8] = logits[:, :8] + move_bias
        value = critic_head(hidden)
        return logits, value

    def forward(self, vector_obs, map_obs, policy_mode=None, inference=False):
        vector_hidden = self._encode_vector_obs(vector_obs)

        x = self.map_stem(map_obs)
        x = self.map_stage1(x)
        x = self.map_stage2(x)
        x = self.map_stage3(x)

        pooled = self.map_pool(x).flatten(1)
        map_hidden = self.map_fc(pooled)

        hidden = torch.cat([vector_hidden, map_hidden], dim=1)
        hidden = self.fusion(hidden)

        escape_logits, escape_value = self._apply_policy_head(
            hidden,
            map_hidden,
            self.escape_actor_head,
            self.escape_critic_head,
            self.escape_move_bias_head,
        )
        treasure_logits, treasure_value = self._apply_policy_head(
            hidden,
            map_hidden,
            self.treasure_actor_head,
            self.treasure_critic_head,
            self.treasure_move_bias_head,
        )

        if policy_mode is None:
            policy_mode = torch.full(
                (hidden.size(0),),
                Config.ESCAPE_POLICY_MODE,
                dtype=torch.long,
                device=hidden.device,
            )
        else:
            policy_mode = policy_mode.view(-1).long()

        treasure_mask = (policy_mode == Config.TREASURE_POLICY_MODE).view(-1, 1)
        logits = torch.where(treasure_mask, treasure_logits, escape_logits)
        value = torch.where(treasure_mask, treasure_value, escape_value)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
