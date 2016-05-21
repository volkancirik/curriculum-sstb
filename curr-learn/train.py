# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout
from keras.layers import recurrent
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l1l2

import json, time, datetime, os, sys
import numpy as np
import cPickle as pickle
import random

from keras.layers import dropoutrnn

from prepare_data import prepare_data
from utils import get_sa, get_embeddings
from buckets import distribute_buckets
"""
train a sentiment analysis model based on lstm-family networks
"""
UNIT = { 'gru' : dropoutrnn.DropoutGRU, 'lstm' : dropoutrnn.DropoutLSTM , 'bilstm' : dropoutrnn.DropoutbiLSTM}

parser = get_sa()
p = parser.parse_args()

# Parameters for the model and dataset
TIMESTAMP = "_".join(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S').split())
SEED = p.seed
randomize = False
if SEED != 0:
	random.seed(SEED)
	randomize = True

RNN = UNIT[p.unit]
EPOCH = p.n_epochs
LAYERS = p.layers
HIDDEN_SIZE = p.n_hidden
BATCH_SIZE = p.batch_size
PATIENCE = p.patience
DROPOUT = p.dropout
weight_decay = 0.0001
EMBEDDING = True
PREFIX = 'exp/'+p.prefix + '/'
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'U'+ p.unit + '_L' + str(LAYERS) + '_H' + str(HIDDEN_SIZE) + '_EMB' + str(EMBEDDING) + '_Dr' + str(DROPOUT) + '_S' + str(p.sentence) + '_SEED'+ str(SEED) + '.' + TIMESTAMP

X_tr, Y_tr, X_val, Y_val, X_test, Y_test, dicts, [length_tr,_,_] = prepare_data('ss', { 'n' : 20, 'max_len' : 10, 'plus_sign' : True}, clip = 1)
b_X_tr, b_Y_tr = distribute_buckets(length_tr, [X_tr], [Y_tr], step_size = 1, x_set = set([0]), y_set = set())

print('building model...')

VOCAB = len(dicts['word_idx'])
EMB_HIDDEN_SIZE = HIDDEN_SIZE
if p.pretrained != '':
	embedding_weights,word_idx, idx_word = get_embeddings(dicts['word_idx'],dicts['idx_word'], wvec = p.pretrained)
	dicts['word_idx'] = word_idx
	dicts['idx_word'] = idx_word
	VOCAB = embedding_weights.shape[0]
	EMB_HIDDEN_SIZE = embedding_weights.shape[1]

model = Sequential()
if EMBEDDING:
	if p.pretrained != "":
		model.add(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True,  input_shape = (None,), weights=[embedding_weights]))
	else:
		model.add(Embedding(VOCAB,HIDDEN_SIZE,mask_zero=True, input_shape = (None,)))
	model.add(RNN(EMB_HIDDEN_SIZE, output_dim = HIDDEN_SIZE, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT ,return_sequences = (LAYERS >= 2)))
else:
	model.add(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay),
               b_regularizer=l2(weight_decay), dropout = DROPOUT , return_sequences = (LAYERS >= 2), input_shape = (None,VOCAB)))

for layer in xrange(LAYERS-2):
	model.add(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT, return_sequences = True))

if LAYERS >= 2:
	model.add(RNN(HIDDEN_SIZE , output_dim = HIDDEN_SIZE, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT ,return_sequences = False))

#model.add(Dense(5, activation = 'softmax', W_regularizer = l1l2(l1 = 0.0001, l2 = 0.0001)))
model.add(Dense(1, activation = 'relu', W_regularizer = l1l2(l1 = 0.0001, l2 = 0.0001)))


optimizer = RMSprop(clipnorm = 10)
print('compiling model...')
#model.compile(loss='binary_crossentropy', optimizer = optimizer)
model.compile(loss='mse', optimizer = optimizer)


#print("saving everything...")
train_history = {'loss' : [], 'val_loss' : [], 'acc' : [], 'val_acc' : []}
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
    json.dump(model.to_json(), outfile)
pickle.dump({'train_history' : train_history, 'dicts' : dicts,'result' : [], 'X_test' : X_test, 'Y_test' : Y_test},open(PREFIX + FOOTPRINT + '.meta', 'w'))

print("training model with {} parameters...".format(model.get_n_params()))
NB = len(b_X_tr)
best_val_acc = 0
pat = 0
BLIST = range(NB)
for iteration in xrange(EPOCH):
	print('_' * 50)
	train_history['loss'] += [0]

	if randomize:
		random.shuffle(BLIST)
	for j in BLIST:
		[X_train] = b_X_tr[j]
		[Y_train] = b_Y_tr[j]

		if len(X_train) == 0:
			continue

		print('iteration {}/{} bucket {}/{}'.format(iteration+1,EPOCH, j+1,NB))

		eh = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, nb_epoch = 1)
		for key in ['loss']:
			train_history[key][-1] += eh.history[key][0]

	vl = model.evaluate(X_val,Y_val, batch_size = BATCH_SIZE, verbose = True)#, show_accuracy = True)
	train_history['val_loss'] += [vl]
	print(train_history['val_loss'])
#	train_history['val_acc'] += [vl[1]]

#	print("VAL {} best VAL {} no improvement in {}".format(train_history['val_acc'][-1],best_val_acc,pat))

	# if train_history['val_acc'][-1] <= best_val_acc:
	# 	pat += 1
	# else:
	# 	pat = 0
	# 	best_val_acc = train_history['val_acc'][-1]
	# 	model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
	# if pat == PATIENCE:
	# 	break

model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
print('testing...')
result = model.evaluate(X_test, Y_test, batch_size = BATCH_SIZE, verbose = 1, show_accuracy=True)

pickle.dump({'train_history' : train_history, 'dicts' : dicts,'result' : result, 'X_test' : X_test, 'Y_test' : Y_test},open(PREFIX + FOOTPRINT + '.meta', 'w'))
print("Test accuracy: ",result,'\nDONE!')
