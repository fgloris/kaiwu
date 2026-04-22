# 基于你当前代码的 V6 定点增强说明

这次不是重写，而是直接在你当前这份 V5 风格代码上补强。

## 主要修改文件

- `agent_ppo/conf/conf.py`
- `agent_ppo/feature/preprocessor.py`
- `agent_ppo/feature/definition.py`
- `agent_ppo/model/model.py`
- `agent_ppo/agent.py`
- `agent_ppo/workflow/train_workflow.py`
- `agent_ppo/conf/monitor_builder.py`
- `agent_ppo/conf/train_env_conf.toml`

## 这次重点改了什么

### 1）特征：补了“记忆”和“探索”
- 新增宝箱 / buff 的 episode 内缓存。
- 当前视野外但以前见过的目标，仍然能作为候选目标进入特征。
- 新增 10 步前位移特征 `past_dx / past_dz`。
- 新增当前位置的新颖度 `explore_novelty`。
- local patch 从 3 通道改成 4 通道，额外加入访问热度 / 新颖度通道。

### 2）奖励：补了反磨蹭和探索奖励
- 保留你原来 V5 里的：
  - survive
  - treasure / buff / dist shaping
  - corridor / deadend / trap / danger
  - flash quality
- 新增：
  - 首次看到宝箱奖励
  - 新区域探索奖励
  - 10 步 anti-stall 惩罚
  - 接近宝箱但同时把自己送进危险区的惩罚

### 3）算法样本：修了 truncated bootstrap
- 现在如果是 `terminated=True`，最后一步 `done=1`，`next_value=0`。
- 如果是到 `max_step` 截断，最后一步不再强行当作真正终局，而是使用 value bootstrap。
- 这能减少 2000 步截断时对 PPO value 学习的干扰。

### 4）workflow：补了 train / val split
- 训练图：`1~8`
- 验证图：`9~10`
- 每 `10` 个 episode 跑 `1` 局 val。
- monitor 里新增 train/val 总分、步数分、宝箱分、生存率、泛化差距。

## 建议你先看哪些监控

优先看这几项：
- `train_total_score`
- `val_total_score`
- `generalization_gap_total`
- `train_step_score`
- `train_treasure_score`
- `val_win_rate`

## 我对这版的预期

这版最主要的提升方向不是“前期疯狂多吃箱子”，而是：
1. 减少原地磨蹭 / 绕圈 / 无效移动；
2. 让 agent 更稳定地记住见过但暂时没吃的资源；
3. 让闪现更偏向“脱险或高价值位移”；
4. 减少 train 看起来上涨但 val 不涨的问题；
5. 避免 2000 步截断把 value 学歪。

## 训练观察建议

如果你后面发现：
- 宝箱还是偏少：优先微增 `treasure_approach_reward / first_seen_treasure_reward / exploration_reward`
- 后期暴毙：优先再加 `danger_penalty / trap_penalty / flash_reward`
- train 高但 val 低：优先检查 curriculum 和 memory 特征是否过拟合开放图

