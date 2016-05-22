# -*- coding: utf-8 -*-
from __future__ import print_function

import json, time, datetime, os, sys
import numpy as np
import cPickle as pickle
import random

from prepare_data import prepare_data
from utils import get_sa, get_embeddings
from buckets import distribute_buckets
from get_model import get_model

from regime import vanilla, onepass_curr, curriculum
"""
train a sentiment analysis model based on lstm-family networks
"""


parser = get_sa()
p = parser.parse_args()

SEED = p.seed
EPOCH = p.n_epochs
BATCH_SIZE = p.batch_size
REGIME = p.regime
random.seed(SEED)

pd_args = { 'sa' : { 'prefix' : '../data/', 'train_f' : 'train_', 'val_f' : 'dev_root.txt', 'test_f' : 'test_root.txt'}, 'ss' : {'n' : 1024, 'min_len' : 2, 'max_len' : 20, 'plus_sign' : True} }
X_tr, Y_tr, X_val, Y_val, X_test, Y_test, dicts, [length_tr,_,_] = prepare_data(p.dataset, pd_args[p.dataset] , root = p.root, clip = 1)

b_X_tr, b_Y_tr = distribute_buckets(length_tr, [X_tr], [Y_tr], step_size = 1, x_set = set([0]), y_set = set())

embedding_weights = None
if p.pretrained != '':
	embedding_weights,word_idx, idx_word = get_embeddings(dicts['word_idx'],dicts['idx_word'], wvec = p.pretrained)
	dicts['word_idx'] = word_idx
	dicts['idx_word'] = idx_word
else:
	print("learning embeddings from data")

model, PREFIX, FOOTPRINT, RANDOMIZE = get_model(p, dicts, embedding_weights = embedding_weights)

with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
    json.dump(model.to_json(), outfile)
pickle.dump({'dicts' : dicts,'result' : [], 'X_test' : X_test, 'Y_test' : Y_test},open(PREFIX + FOOTPRINT + '.meta', 'w'))

print("training model with {} parameters...".format(model.get_n_params()))

NB = len(b_X_tr)
regimes = {'vanilla' : vanilla, 'onepass' : onepass_curr, 'curriculum' : curriculum}
train_history = regimes[REGIME]([b_X_tr, b_Y_tr, X_val, Y_val], model, EPOCH, RANDOMIZE, BATCH_SIZE, p.patience, NB, PREFIX, FOOTPRINT, p.dataset == 'sa')

print('testing...')
model.load_weights(PREFIX + FOOTPRINT + '.model')
result = model.evaluate(X_test, Y_test, batch_size = BATCH_SIZE, verbose = 1, show_accuracy = True)

pickle.dump({'train_history' : train_history, 'dicts' : dicts,'result' : result, 'X_test' : X_test, 'Y_test' : Y_test},open(PREFIX + FOOTPRINT + '.meta', 'w'))
print("Test result: ",result,'\nDONE!')
