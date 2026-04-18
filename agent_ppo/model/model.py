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
        self.model_name = "gorge_chase_cnn_gru_v1"
        self.device = device

        vector_dim = Config.VECTOR_FEATURE_LEN
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        rnn_hidden_size = Config.RNN_HIDDEN_SIZE

        self.vec_dropout_p = 0.10
        self.map_dropout_p = 0.10
        self.fusion_dropout_p = 0.15
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
            ResidualBlock(32, dropout_p=self.map_dropout_p),
            nn.MaxPool2d(2),
        )

        self.map_stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.map_dropout_p),
            ResidualBlock(64, dropout_p=self.map_dropout_p),
            ResidualBlock(64, dropout_p=self.map_dropout_p),
            nn.MaxPool2d(2),
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

        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=rnn_hidden_size,
            num_layers=Config.RNN_NUM_LAYERS,
            batch_first=True,
        )
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

        self.actor_head = nn.Sequential(
            make_fc_layer(rnn_hidden_size, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, action_num),
        )

        self.critic_head = nn.Sequential(
            make_fc_layer(rnn_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(self.head_dropout_p),
            make_fc_layer(256, 128),
            nn.ReLU(),
            nn.Dropout(self.head_dropout_p),
            make_fc_layer(128, value_num),
        )

        self.move_bias_head = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
            make_fc_layer(64, 8),
        )

    def _encode_obs(self, vector_obs, map_obs):
        vector_hidden = self.vector_encoder(vector_obs)

        x = self.map_stem(map_obs)
        x = self.map_stage1(x)
        x = self.map_stage2(x)
        x = self.map_stage3(x)

        pooled = self.map_pool(x).flatten(1)
        map_hidden = self.map_fc(pooled)

        hidden = torch.cat([vector_hidden, map_hidden], dim=1)
        hidden = self.fusion(hidden)
        return hidden, map_hidden

    def forward(self, vector_obs, map_obs, hidden_state=None, inference=False):
        fused_hidden, map_hidden = self._encode_obs(vector_obs, map_obs)

        if inference:
            rnn_in = fused_hidden.unsqueeze(1)
            rnn_out, next_hidden_state = self.rnn(rnn_in, hidden_state)
            rnn_hidden = rnn_out.squeeze(1)
        else:
            rnn_in = fused_hidden.unsqueeze(0)
            rnn_out, next_hidden_state = self.rnn(rnn_in, hidden_state)
            rnn_hidden = rnn_out.squeeze(0)

        logits = self.actor_head(rnn_hidden)
        move_bias = self.move_bias_head(map_hidden)
        logits[:, :8] = logits[:, :8] + move_bias

        value = self.critic_head(rnn_hidden)
        return logits, value, next_hidden_state

    def get_initial_state(self, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.zeros(
            Config.RNN_NUM_LAYERS,
            batch_size,
            Config.RNN_HIDDEN_SIZE,
            device=device,
            dtype=torch.float32,
        )

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
