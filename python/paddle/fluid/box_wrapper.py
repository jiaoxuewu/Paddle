#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: jiaoxuewu@baidu.com (Xuewu Jiao)
@Copyright: Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
@Desc:
@ChangeLog:
"""
from . import core

__all__ = ['BoxWrapper']

class BoxWrapper(object):
    """
    box wrapper class
    """

    def __init__(self):
        self.box_wrapper = core.BoxWrapper()


    def save_model(self):
        self.box_wrapper.save_model()
