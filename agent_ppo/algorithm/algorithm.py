#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import time
import torch
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        self.ppo_epoch = Config.PPO_EPOCH
        self.minibatch_size = Config.MINIBATCH_SIZE

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        if not list_sample_data:
            return

        obs = torch.stack([f.obs for f in list_sample_data]).float().to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).float().to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).float().to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).float().to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).float().to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).float().to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).float().to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).float().to(self.device)

        advantage = (advantage - advantage.mean()) / (advantage.std() + Config.ADV_NORM_EPS)

        batch_size = obs.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)
        last_total_loss = None
        last_info = None

        for _ in range(self.ppo_epoch):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_idx = indices[start:end]

                self.model.set_train_mode()
                self.optimizer.zero_grad()

                logits, value_pred = self.model(obs[mb_idx])
                total_loss, info_list = self._compute_loss(
                    logits=logits,
                    value_pred=value_pred,
                    legal_action=legal_action[mb_idx],
                    old_action=act[mb_idx],
                    old_prob=old_prob[mb_idx],
                    advantage=advantage[mb_idx],
                    old_value=old_value[mb_idx],
                    reward_sum=reward_sum[mb_idx],
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
                self.optimizer.step()

                last_total_loss = total_loss
                last_info = info_list
                self.train_step += 1

        now = time.time()
        if last_total_loss is not None and now - self.last_report_monitor_time >= 60:
            with torch.no_grad():
                approx_kl = last_info[3].item()
                clip_frac = last_info[4].item()
                entropy_val = last_info[2].item()
                results = {
                    "total_loss": round(last_total_loss.item(), 4),
                    "value_loss": round(last_info[0].item(), 4),
                    "policy_loss": round(last_info[1].item(), 4),
                    "entropy_loss": round(entropy_val, 4),
                    "reward": round(reward.mean().item(), 4),
                    "approx_kl": round(approx_kl, 6),
                    "clip_frac": round(clip_frac, 4),
                }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} entropy:{results['entropy_loss']} "
                f"kl:{results['approx_kl']} clip_frac:{results['clip_frac']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
    ):
        prob_dist = self._masked_softmax(logits, legal_action)
        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True).clamp_min(1e-8)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp_min(1e-8)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)

        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.maximum(torch.square(tdret - vp), torch.square(tdret - value_clip)).mean()

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp_min(1e-8))).sum(1).mean()
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        with torch.no_grad():
            log_ratio = torch.log(new_prob) - torch.log(old_action_prob)
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean()

        return total_loss, [value_loss, policy_loss, entropy_loss, approx_kl, clip_frac]

    def _masked_softmax(self, logits, legal_action):
        mask = legal_action.float()
        valid_count = mask.sum(dim=1, keepdim=True)
        fallback_mask = torch.ones_like(mask)
        mask = torch.where(valid_count > 0, mask, fallback_mask)
        masked_logits = logits.masked_fill(mask < 0.5, -1e9)
        return torch.nn.functional.softmax(masked_logits, dim=1)
