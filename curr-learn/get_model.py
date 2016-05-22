# -*- coding: utf-8 -*-
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

from utils import get_embeddings
UNIT = { 'gru' : dropoutrnn.DropoutGRU, 'lstm' : dropoutrnn.DropoutLSTM , 'bilstm' : dropoutrnn.DropoutbiLSTM, 'bigru' : dropoutrnn.DropoutbiGRU, 'dan' : []}

def dan(model, RNN, LAYERS, EMB_HIDDEN_SIZE, HIDDEN_SIZE, weight_decay, DROPOUT):
	from keras.layers.averagelayer import Average
	model.add(Average())
	for layer in xrange(LAYERS):
		model.add(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.0001, l2 = 0.0001)))
		model.add(Dropout(DROPOUT))
	return model

def uni_dir(model, RNN, LAYERS, EMB_HIDDEN_SIZE, HIDDEN_SIZE, weight_decay, DROPOUT):
	model.add(RNN(EMB_HIDDEN_SIZE, output_dim = HIDDEN_SIZE, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT ,return_sequences = (LAYERS >= 2)))

	for layer in xrange(LAYERS-2):
		model.add(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT, return_sequences = True))

	if LAYERS >= 2:
		model.add(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT ,return_sequences = False))
	return model

def bi_dir(model, RNN, LAYERS, EMB_HIDDEN_SIZE, HIDDEN_SIZE, weight_decay, DROPOUT):

	model.add(RNN(EMB_HIDDEN_SIZE, output_dim = HIDDEN_SIZE * 2, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT ,return_sequences = (LAYERS >= 2)))

	for layer in xrange(LAYERS-2):
		model.add(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE * 2, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT, return_sequences = True))

	if LAYERS >= 2:
		model.add(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE * 2, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), dropout = DROPOUT ,return_sequences = False))

	return model

def get_model(p, dicts, embedding_weights = None):
	MODEL = {'lstm' : uni_dir, 'gru' : uni_dir, 'bilstm' : bi_dir, 'bigru' : bi_dir, 'dan' : dan}
	LOSS = {'ss' : 'mse', 'sa' : 'categorical_crossentropy'}
	RNN = UNIT[p.unit]

	LAYERS = p.layers
	HIDDEN_SIZE = p.n_hidden

	DATASET = p.dataset
	ROOT = p.root
	PATIENCE = p.patience
	DROPOUT = p.dropout
	weight_decay = 0.0001

	PREFIX = 'exp/'+p.prefix + '/'
	os.system('mkdir -p '+PREFIX)
	TIMESTAMP = "_".join(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S').split())
	SEED = p.seed
	RANDOMIZE = False
	if SEED != 0:
		RANDOMIZE = True

	FOOTPRINT = 'U'+ p.unit + '_L' + str(LAYERS) + '_H' + str(HIDDEN_SIZE) + '_Dr' + str(DROPOUT) +  '_D' + DATASET  + '_SEED'+ str(SEED) + '_ROOT' + ROOT + '_REG' + p.regime + '.' + TIMESTAMP
	VOCAB = len(dicts['word_idx']) + 1
	EMB_HIDDEN_SIZE = HIDDEN_SIZE

	model = Sequential()
	if embedding_weights != None:
		VOCAB = embedding_weights.shape[0]
		EMB_HIDDEN_SIZE = embedding_weights.shape[1]
		model.add(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True,  input_shape = (None,), weights=[embedding_weights]))
	else:
		model.add(Embedding(VOCAB,HIDDEN_SIZE,mask_zero=True, input_shape = (None,)))

	model = MODEL[p.unit](model, RNN, LAYERS, EMB_HIDDEN_SIZE, HIDDEN_SIZE, weight_decay, DROPOUT)
	if DATASET == 'ss':
		model.add(Dense(1, activation = 'relu', W_regularizer = l1l2(l1 = 0.0001, l2 = 0.0001)))
	elif DATASET == 'sa':
		model.add(Dense(5, activation = 'softmax', W_regularizer = l1l2(l1 = 0.0001, l2 = 0.0001)))
	else:
		raise NotImplementedError()

	optimizer = RMSprop(clipnorm = 10)
#	print('compiling model...')

	model.compile(loss=LOSS[DATASET], optimizer = optimizer)
	return model, PREFIX, FOOTPRINT, RANDOMIZE

