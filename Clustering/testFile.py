#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:43:08 2021

@author: marielb
"""


import os


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
myfile = os.path.basename(__file__)
myfile_path = os.path.join(ROOT_PATH, myfile)

print(ROOT_PATH)
print(myfile)
print(myfile_path)