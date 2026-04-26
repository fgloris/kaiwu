#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()

        .add_group(
            group_name="学习率指标",
            group_name_en="learning_rate",
        )
        .add_panel(
            name="episode_cnt",
            name_en="episode_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="episode_cnt",
            expr="avg(episode_cnt{})",
        )
        .add_metric(
            metrics_name="lr_warmup_episodes",
            expr="avg(lr_warmup_episodes{})",
        )
        .add_metric(
            metrics_name="lr_cosine_end_episode",
            expr="avg(lr_cosine_end_episode{})",
        )
        .end_panel()
        .add_panel(
            name="learning_rate",
            name_en="learning_rate",
            type="line",
        )
        .add_metric(
            metrics_name="learning_rate",
            expr="avg(learning_rate{})",
        )
        .add_metric(
            metrics_name="peak_learning_rate",
            expr="avg(peak_learning_rate{})",
        )
        .add_metric(
            metrics_name="min_learning_rate",
            expr="avg(min_learning_rate{})",
        )
        .end_panel()
        .end_group()

        .add_group(
            group_name="环境指标",
            group_name_en="env",
        )
        .add_panel(
            name="训练平均得分",
            name_en="train_avg_scores",
            type="line",
        )
        .add_metric(metrics_name="train_avg_total_score", expr="avg(train_avg_total_score{})")
        .add_metric(metrics_name="train_avg_treasure_score", expr="avg(train_avg_treasure_score{})")
        .add_metric(metrics_name="train_avg_step_score", expr="avg(train_avg_step_score{})")
        .end_panel()
        .add_panel(
            name="训练最低得分",
            name_en="train_min_scores",
            type="line",
        )
        .add_metric(metrics_name="train_min_total_score", expr="avg(train_min_total_score{})")
        .add_metric(metrics_name="train_min_treasure_score", expr="avg(train_min_treasure_score{})")
        .add_metric(metrics_name="train_min_step_score", expr="avg(train_min_step_score{})")
        .end_panel()
        .add_panel(
            name="测试地图1_2得分",
            name_en="eval_map_1_2_scores",
            type="line",
        )
        .add_metric(metrics_name="eval_12_total_score", expr="avg(eval_12_total_score{})")
        .add_metric(metrics_name="eval_12_treasure_score", expr="avg(eval_12_treasure_score{})")
        .add_metric(metrics_name="eval_12_step_score", expr="avg(eval_12_step_score{})")
        .add_metric(metrics_name="eval_12_min_total_score", expr="avg(eval_12_min_total_score{})")
        .add_metric(metrics_name="eval_12_min_treasure_score", expr="avg(eval_12_min_treasure_score{})")
        .add_metric(metrics_name="eval_12_min_step_score", expr="avg(eval_12_min_step_score{})")
        .end_panel()
        .add_panel(
            name="测试地图9_10得分",
            name_en="eval_map_9_10_scores",
            type="line",
        )
        .add_metric(metrics_name="eval_910_total_score", expr="avg(eval_910_total_score{})")
        .add_metric(metrics_name="eval_910_treasure_score", expr="avg(eval_910_treasure_score{})")
        .add_metric(metrics_name="eval_910_step_score", expr="avg(eval_910_step_score{})")
        .add_metric(metrics_name="eval_910_min_total_score", expr="avg(eval_910_min_total_score{})")
        .add_metric(metrics_name="eval_910_min_treasure_score", expr="avg(eval_910_min_treasure_score{})")
        .add_metric(metrics_name="eval_910_min_step_score", expr="avg(eval_910_min_step_score{})")
        .end_panel()
        .end_group()

        .add_group(
            group_name="穿怪闪现",
            group_name_en="through_monster_flash",
        )
        .add_panel(
            name="使用次数",
            name_en="through_flash_usage",
            type="line",
        )
        .add_metric(metrics_name="through_flash_avg_count", expr="avg(through_flash_avg_count{})")
        .add_metric(metrics_name="through_flash_legal_avg_count", expr="avg(through_flash_legal_avg_count{})")
        .end_panel()
        .add_panel(
            name="闪后存活",
            name_en="through_flash_survival",
            type="line",
        )
        .add_metric(metrics_name="through_flash_avg_survival", expr="avg(through_flash_avg_survival{})")
        .end_panel()
        .end_group()

        .add_group(
            group_name="reward_breakdown",
            group_name_en="reward_breakdown",
        )
        .add_panel(
            name="Reward Vector",
            name_en="reward_vector",
            type="line",
        )
        .add_metric(metrics_name="r_score_gain_sum", expr="avg(r_score_gain_sum{})")
        .add_metric(metrics_name="r_survival_gain_sum", expr="avg(r_survival_gain_sum{})")
        .add_metric(metrics_name="r_monster_los_break_sum", expr="avg(r_monster_los_break_sum{})")
        .add_metric(metrics_name="r_flash_sum", expr="avg(r_flash_sum{})")
        .add_metric(metrics_name="r_wall_penalty_sum", expr="avg(r_wall_penalty_sum{})")
        .add_metric(metrics_name="r_abb_penalty_sum", expr="avg(r_abb_penalty_sum{})")
        .add_metric(metrics_name="r_danger_penalty_sum", expr="avg(r_danger_penalty_sum{})")
        .add_metric(metrics_name="r_treasure_dist_sum", expr="avg(r_treasure_dist_sum{})")
        .add_metric(metrics_name="r_buff_dist_sum", expr="avg(r_buff_dist_sum{})")
        .add_metric(metrics_name="r_buff_pick_sum", expr="avg(r_buff_pick_sum{})")
        .add_metric(metrics_name="r_monster_dist_sum", expr="avg(r_monster_dist_sum{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
