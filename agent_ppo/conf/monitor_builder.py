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

        # 新增：环境测试指标
        .add_group(
            group_name="环境指标",
            group_name_en="env",
        )
        .add_panel(
            name="测试环境得分",
            name_en="eval_env_scores",
            type="line",
        )
        .add_metric(
            metrics_name="eval_total_score",
            expr="avg(eval_total_score{})",
        )
        .add_metric(
            metrics_name="eval_treasure_score",
            expr="avg(eval_treasure_score{})",
        )
        .add_metric(
            metrics_name="eval_step_score",
            expr="avg(eval_step_score{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
