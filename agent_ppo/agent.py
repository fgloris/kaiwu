#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(feature=list(feature), legal_action=legal_action)
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
        logits, value, prob = self._run_model(feature, legal_action)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)
        return [ActData(action=[action], d_action=[d_action], prob=list(prob), value=value)]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)
        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)
        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        legal_action = np.asarray(legal_action, dtype=np.float32)
        if legal_action.sum() <= 0:
            legal_action = np.ones_like(legal_action, dtype=np.float32)
        masked = np.where(legal_action > 0.5, input_hidden, -1e9)
        masked = masked - np.max(masked, keepdims=True)
        exp_logits = np.exp(np.clip(masked, -50.0, 50.0)) * legal_action
        denom = np.sum(exp_logits, keepdims=True)
        if float(denom) <= 0:
            return legal_action / np.sum(legal_action)
        return exp_logits / denom

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
