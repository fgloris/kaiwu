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
            group_name="奖励分解",
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
