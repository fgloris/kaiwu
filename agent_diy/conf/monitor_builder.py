#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()
    config_dict = (
        monitor.title("峡谷追猎-DIY规划")
        .add_group(group_name="运行指标", group_name_en="runtime")
        .add_panel(name="累计回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="总分", name_en="total_score", type="line")
        .add_metric(metrics_name="total_score", expr="avg(total_score{})")
        .add_metric(metrics_name="avg_total_score", expr="avg(avg_total_score{})")
        .end_panel()
        .add_panel(name="步数分", name_en="step_score", type="line")
        .add_metric(metrics_name="step_score", expr="avg(step_score{})")
        .add_metric(metrics_name="avg_step_score", expr="avg(avg_step_score{})")
        .end_panel()
        .add_panel(name="A星路径长度", name_en="path_len", type="line")
        .add_metric(metrics_name="path_len", expr="avg(path_len{})")
        .end_panel()
        .add_panel(name="目标点评分", name_en="target_score", type="line")
        .add_metric(metrics_name="target_score", expr="avg(target_score{})")
        .end_panel()
        .add_panel(name="边界点与簇", name_en="frontier_cluster", type="line")
        .add_metric(metrics_name="frontier_count", expr="avg(frontier_count{})")
        .add_metric(metrics_name="cluster_count", expr="avg(cluster_count{})")
        .end_panel()
        .add_panel(name="Fallback比例", name_en="fallback_rate", type="line")
        .add_metric(metrics_name="fallback_rate", expr="avg(fallback_rate{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
