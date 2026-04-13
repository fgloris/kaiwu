#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


class Algorithm:
    def __init__(self, model=None, optimizer=None, scheduler=None, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.monitor = monitor

    def learn(self, list_sample_data):
        return None
