#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : mclane.py

from datasets.BaseDataset import BaseDataset


class AniSeg(BaseDataset):
    @classmethod
    def get_class_colors(*args):
        return [[0, 0, 0],
                [255, 0, 0], 
                [255, 255, 0], 
                [255, 97, 0], 
                [245, 222, 179],
                [218, 165, 105], 
                [127, 255, 0], 
                [0, 255, 0],
                [8, 46, 84],
                [64, 224, 208], 
                [176, 226, 255], 
                [0, 139, 139],
                [144, 238, 144]]

    @classmethod
    def get_class_names(*args):
        return ['00_unlabeled','01_hair', '02_face', '03_eyes',"04_assesories",
                '05_ears',
                '06_torso', '07_torso_wearing', '08_arms', '09_hands',
                '10_legs',
                '11_feet', '12_legs_wearing/decoration']



