#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()
    config_dict = (
        monitor.title("峡谷追猎 PPO Strong")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="累积回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
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
        .add_panel(name="熵", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .add_panel(name="KL", name_en="approx_kl", type="line")
        .add_metric(metrics_name="approx_kl", expr="avg(approx_kl{})")
        .end_panel()
        .add_panel(name="裁剪比例", name_en="clipfrac", type="line")
        .add_metric(metrics_name="clipfrac", expr="avg(clipfrac{})")
        .end_panel()
        .add_panel(name="最大动作概率", name_en="max_prob", type="line")
        .add_metric(metrics_name="max_prob", expr="avg(max_prob{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
