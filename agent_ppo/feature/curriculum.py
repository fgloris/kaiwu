from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    stage: int
    name: str
    episode_end: int
    treasure_score_gain: float
    survival: float
    los_break: float
    flash: float
    wall_penalty: float
    abb_penalty: float
    danger_penalty: float
    treasure_dist: float
    buff_dist: float
    buff_pick: float
    monster_dist: float
    survival_multiplier: float


# Single reward setting, using the previous stage-3 "advanced" parameters.
REWARD_CONFIG = RewardConfig(
    stage=3,
    name="advanced",
    episode_end=-1,
    treasure_score_gain=0.25,
    survival=0.40,
    los_break=0.40,
    flash=0.25,
    wall_penalty=0.10,
    abb_penalty=0.15,
    danger_penalty=0.35,
    treasure_dist=0.30,
    buff_dist=0.30,
    buff_pick=0.70,
    monster_dist=1.00,
    survival_multiplier=1.10,
)
