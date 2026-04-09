#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

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
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        self.entropy_coef = Config.BETA_START
        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        obs = torch.stack([torch.as_tensor(f.obs, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([torch.as_tensor(f.legal_action, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        act = torch.stack([torch.as_tensor(f.act, dtype=torch.float32) for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([torch.as_tensor(f.prob, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        reward = torch.stack([torch.as_tensor(f.reward, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        advantage = torch.stack([torch.as_tensor(f.advantage, dtype=torch.float32) for f in list_sample_data]).to(self.device).view(-1, 1)
        old_value = torch.stack([torch.as_tensor(f.value, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([torch.as_tensor(f.reward_sum, dtype=torch.float32) for f in list_sample_data]).to(self.device)

        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + Config.EPS)
        old_log_prob = self._action_log_prob(old_prob, act)

        batch_size = obs.shape[0]
        mini_batch_size = min(Config.MINI_BATCH_SIZE, batch_size)
        epoch_stats = []

        for epoch in range(Config.PPO_EPOCHS):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                logits, value_pred = self.model(obs[idx])
                total_loss, stats = self._compute_loss(
                    logits=logits,
                    value_pred=value_pred,
                    legal_action=legal_action[idx],
                    action=act[idx],
                    old_log_prob=old_log_prob[idx],
                    advantage=advantage[idx],
                    old_value=old_value[idx],
                    reward_sum=reward_sum[idx],
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
                self.optimizer.step()
                epoch_stats.append(stats)
                self.train_step += 1
                if stats["approx_kl"] > Config.TARGET_KL:
                    break
            if epoch_stats and epoch_stats[-1]["approx_kl"] > Config.TARGET_KL:
                break

        if not epoch_stats:
            return

        keys = list(epoch_stats[0].keys())
        avg_stats = {k: float(np.mean([s[k] for s in epoch_stats])) for k in keys}
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            log_str = (
                f"[train] total_loss:{avg_stats['total_loss']:.4f} policy_loss:{avg_stats['policy_loss']:.4f} "
                f"value_loss:{avg_stats['value_loss']:.4f} entropy:{avg_stats['entropy']:.4f} "
                f"approx_kl:{avg_stats['approx_kl']:.5f} clipfrac:{avg_stats['clipfrac']:.3f} "
                f"adv_std:{avg_stats['adv_std']:.4f} max_prob:{avg_stats['max_prob']:.3f} reward:{reward.mean().item():.4f}"
            )
            self.logger.info(log_str)
            if self.monitor:
                self.monitor.put_data(
                    {
                        os.getpid(): {
                            "total_loss": round(avg_stats["total_loss"], 4),
                            "value_loss": round(avg_stats["value_loss"], 4),
                            "policy_loss": round(avg_stats["policy_loss"], 4),
                            "entropy_loss": round(avg_stats["entropy"], 4),
                            "approx_kl": round(avg_stats["approx_kl"], 5),
                            "clipfrac": round(avg_stats["clipfrac"], 4),
                            "max_prob": round(avg_stats["max_prob"], 4),
                            "reward": round(reward.mean().item(), 4),
                        }
                    }
                )
            self.last_report_monitor_time = now

    def _compute_loss(self, logits, value_pred, legal_action, action, old_log_prob, advantage, old_value, reward_sum):
        prob_dist = self._masked_softmax(logits, legal_action)
        new_log_prob = self._action_log_prob(prob_dist, action)
        log_ratio = new_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantage
        surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_pred_clipped = old_value + (value_pred - old_value).clamp(-self.clip_param, self.clip_param)
        value_loss_unclipped = (value_pred - reward_sum).pow(2)
        value_loss_clipped = (value_pred_clipped - reward_sum).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        entropy = -(prob_dist * torch.log(prob_dist.clamp_min(Config.EPS))).sum(dim=1).mean()
        total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

        clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
        approx_kl = (old_log_prob - new_log_prob).mean().abs().item()
        max_prob = prob_dist.max(dim=1).values.mean().item()

        stats = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "adv_std": advantage.std(unbiased=False).item(),
            "max_prob": max_prob,
        }
        return total_loss, stats

    def _masked_softmax(self, logits, legal_action):
        masked_logits = logits.masked_fill(legal_action < 0.5, -1e9)
        return F.softmax(masked_logits, dim=1)

    def _action_log_prob(self, prob_dist, action):
        one_hot = F.one_hot(action[:, 0].long(), self.label_size).float()
        chosen_prob = (one_hot * prob_dist).sum(dim=1, keepdim=True).clamp_min(Config.EPS)
        return chosen_prob.log()
