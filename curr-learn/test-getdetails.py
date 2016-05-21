# -*- coding: utf-8 -*-
import numpy as np
import theano
from keras.models import Sequential, slice_X
from keras.models import model_from_json
import sys
import json, time, datetime, os
import cPickle as pickle
from collections import defaultdict

prefix = sys.argv[1]

meta = prefix + '.meta'
arch = prefix + '.arch'
model_filename = prefix + '.model'

meta_dict = pickle.load(open(meta))

with open(arch) as json_file:
	architecture = json.load(json_file)
model = model_from_json(architecture)
model.load_weights(model_filename)

print model.evaluate(meta_dict['X_test'], meta_dict['Y_test'], batch_size = 128, verbose = 1, show_accuracy=True)
for layer in model.layers:
	print layer
get_details = theano.function([model.layers[0].input], model.layers[1].get_details(train=False), allow_input_downcast=True)

[outputs, memories, gate_i, gate_f, gate_o] = get_details(meta_dict['X_test'])
print outputs.shape
print memories.shape
print gate_i.shape
print gate_f.shape
print gate_o.shape

print "_" * 50
print outputs[0]
print
print memories[0]
print
print gate_i[0]
print
print gate_f[0]
print
print gate_o[0]

