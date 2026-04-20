from dataclasses import dataclass

CURRICULUM_STAGE_1_END = 800
CURRICULUM_STAGE_2_END = 1800
CURRICULUM_STAGE_3_END = 3000

@dataclass(frozen=True)
class CurriculumRewardConfig:
    stage: int
    name: str
    episode_end: int
    score_gain: float
    survival: float
    los_break: float
    flash: float
    wall_penalty: float
    abb_penalty: float
    exploration: float
    danger_penalty: float
    treasure_dist: float
    buff_dist: float
    buff_pick: float
    monster_dist: float
    survival_multiplier: float


# Reward curriculum table. episode_end is exclusive; -1 means no upper bound.
CURRICULUM_REWARD_CONFIGS = (
    CurriculumRewardConfig(
        stage=1,
        name="intro",
        episode_end=CURRICULUM_STAGE_1_END,
        score_gain=0.50,
        survival=0.15,
        los_break=0.20,
        flash=0.15,
        wall_penalty=0.10,
        abb_penalty=0.10,
        exploration=0.30,
        danger_penalty=0.15,
        treasure_dist=0.80,
        buff_dist=0.40,
        buff_pick=1.00,
        monster_dist=0.80,
        survival_multiplier=0.80,
    ),
    CurriculumRewardConfig(
        stage=2,
        name="growth",
        episode_end=CURRICULUM_STAGE_2_END,
        score_gain=0.35,
        survival=0.25,
        los_break=0.35,
        flash=0.20,
        wall_penalty=0.15,
        abb_penalty=0.15,
        exploration=0.20,
        danger_penalty=0.25,
        treasure_dist=0.50,
        buff_dist=0.35,
        buff_pick=0.80,
        monster_dist=1.20,
        survival_multiplier=1.00,
    ),
    CurriculumRewardConfig(
        stage=3,
        name="advanced",
        episode_end=CURRICULUM_STAGE_3_END,
        score_gain=0.25,
        survival=0.35,
        los_break=0.50,
        flash=0.25,
        wall_penalty=0.20,
        abb_penalty=0.20,
        exploration=0.10,
        danger_penalty=0.35,
        treasure_dist=0.30,
        buff_dist=0.30,
        buff_pick=0.70,
        monster_dist=1.50,
        survival_multiplier=1.10,
    ),
    CurriculumRewardConfig(
        stage=4,
        name="expert",
        episode_end=-1,
        score_gain=0.20,
        survival=0.45,
        los_break=0.60,
        flash=0.30,
        wall_penalty=0.25,
        abb_penalty=0.25,
        exploration=0.05,
        danger_penalty=0.45,
        treasure_dist=0.25,
        buff_dist=0.25,
        buff_pick=0.60,
        monster_dist=1.80,
        survival_multiplier=1.30,
    ),
)