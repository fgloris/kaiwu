# agent_ppo_strong

这是一个可直接替换的更稳 PPO 版本，主要改动：

- done 截断的 GAE
- 优势归一化
- PPO 多 epoch + mini-batch
- 更强的 MLP 编码器
- 更丰富的观测特征：hero / monsters / optional targets / 5x5 map / last action
- 更详细的训练日志
- 更频繁的模型保存

## 切换方法

把 `conf/algo_conf_gorge_chase.toml` 中的 `ppo` 项从：

```toml
actor_agent = "agent_ppo.agent.Agent"
learner_agent = "agent_ppo.agent.Agent"
aisrv_agent = "agent_ppo.agent.Agent"
train_workflow = "agent_ppo.workflow.train_workflow.workflow"
```

改成：

```toml
actor_agent = "agent_ppo_strong.agent.Agent"
learner_agent = "agent_ppo_strong.agent.Agent"
aisrv_agent = "agent_ppo_strong.agent.Agent"
train_workflow = "agent_ppo_strong.workflow.train_workflow.workflow"
```

如果想进一步增加采样稳定性，可以同时把 `conf/configure_app.toml` 里的：

- `train_batch_size` 调到 1024~4096
- `preload_ratio` 降到 0.3~0.5
- `model_file_sync_per_minutes` 保持 1
