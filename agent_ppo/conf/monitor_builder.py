#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Monitor panel configuration builder for Gorge Chase."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder



def build_monitor():
    monitor = MonitorConfigBuilder()
    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(group_name="训练表现", group_name_en="performance")
        .add_panel(name="训练总分", name_en="train_total_score", type="line")
        .add_metric(metrics_name="train_total_score", expr="avg(train_total_score{})")
        .end_panel()
        .add_panel(name="验证总分", name_en="val_total_score", type="line")
        .add_metric(metrics_name="val_total_score", expr="avg(val_total_score{})")
        .end_panel()
        .add_panel(name="训练步数分", name_en="train_step_score", type="line")
        .add_metric(metrics_name="train_step_score", expr="avg(train_step_score{})")
        .end_panel()
        .add_panel(name="训练宝箱分", name_en="train_treasure_score", type="line")
        .add_metric(metrics_name="train_treasure_score", expr="avg(train_treasure_score{})")
        .end_panel()
        .add_panel(name="验证生存率", name_en="val_win_rate", type="line")
        .add_metric(metrics_name="val_win_rate", expr="avg(val_win_rate{})")
        .end_panel()
        .add_panel(name="泛化差距", name_en="generalization_gap_total", type="line")
        .add_metric(metrics_name="generalization_gap_total", expr="avg(generalization_gap_total{})")
        .end_panel()
        .end_group()
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="训练回报", name_en="train_reward", type="line")
        .add_metric(metrics_name="train_reward", expr="avg(train_reward{})")
        .end_panel()
        .add_panel(name="总损失", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
