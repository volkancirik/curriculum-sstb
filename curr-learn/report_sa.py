# -*- coding: utf-8 -*-
import json, time, datetime, os
import cPickle as pickle
import numpy as np

import sys
"""
test LM baseline for CBT
"""

def load_model(path):
	meta = path + '.meta'
	model_filename = path + '.model'

	meta_dict = pickle.load(open(meta))
	return meta_dict

PATH  = sys.argv[1]

meta_dict = load_model(PATH)
try:
	print PATH,":",meta_dict['result'][1]
except:
	pass

