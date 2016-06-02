# -*- coding: utf-8 -*-

import numpy as np
import theano
from keras.models import Sequential, slice_X
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, RepeatVector, Merge, Layer
from keras.regularizers import l1l2

import sys
import json, time, datetime, os
import cPickle as pickle
from collections import defaultdict
from prepare_data import prepare_sa_test, open_file
from buckets import distribute_buckets

conjunctions=[w.strip() for w in 'after, although, as if, as long as, as much as, as soon as, as though, because, before, even, even if, even though, if only, if when, if then, in order that, just as, lest, now, since, now that, now when, once, provided, provided that, rather than, so that, supposing, though, til, unless, until when, whenever, where, whereas, where if, wherever, whether , while'.split(',')]

def analyze_polarity(correct, count, g_idx, p_idx, length_test):

	for idx in xrange(len(g_idx)):
		count[g_idx[idx]] += 1
		if g_idx[idx] == p_idx[idx]:
			correct[g_idx[idx]] += 1.0

def analyze_length(correct, count, g_idx, p_idx, length_test):

	for idx in xrange(len(g_idx)):
		l = length_test[idx]
		if l > 45:
			continue
		for i in xrange(l-2,l+3):
			count[i] += 1
			if g_idx[idx] == p_idx[idx]:
				correct[i] += 1.0

def get_debug_model(model, X):

	get_details = theano.function([model.layers[0].input], model.layers[1].get_details(train=False), allow_input_downcast=True)
	[outputs, memories, gate_i, gate_f, gate_o] = get_details(X)

	HIDDEN_SIZE = outputs.shape[-1]
	debug_model = Sequential()
	debug_model.add(Layer(input_shape = (HIDDEN_SIZE,)))
	debug_model.add(Dense(5, activation = 'softmax', W_regularizer = l1l2(l1 = 0.0001, l2 = 0.0001)))

	debug_model.layers[-1].set_weights( model.layers[-1].get_weights())
	optimizer = RMSprop(clipnorm = 5)
	debug_model.compile(loss='mean_squared_error', optimizer = optimizer)

	return debug_model, [outputs, memories, gate_i, gate_f, gate_o]

def convert2string(instance, meta_dict):
	return [meta_dict['dicts']['idx_word'][w] for w in instance]

def running_decode(model, meta_dict, X,Y, fname = 'TEMP.txt', verbose = True):

	debug_model, gates_output = get_debug_model(model, X)

	o = gates_output[0]

	Xprediction = model.predict(X, verbose = 0, batch_size=16)
	debug_Xprediction = debug_model.predict(o[:,-1,:])
	assert(Xprediction[0][0] == debug_Xprediction[0][0])

	error = 0.0
	L = []

	details = []
	for i in xrange(Xprediction.shape[0]):
		L += [convert2string(X[i], meta_dict)]
		if verbose:
			print("GT: {} PREDICTION : {} INPUT : {}".format(Y,Xprediction[i], " ".join(L[i])))

		for t in xrange(o.shape[1]-1):
			h_t = o[i,t,:]
			o_t = debug_model.predict(np.array([h_t]))

			if verbose:
				print("{} \t {} \t {}".format(o_t[0], np.argmax(o_t[0]), " ".join(L[i][:t+1])))
			details += [(o_t[0], np.argmax(o_t[0]), " ".join(L[i][:t+1]))]

		if verbose:
			print('_'*20)

	return details

def get_s_conjunctions(X_test,Y_test, length_test, fname = '../data/test_root.txt' , lower = True, min_len = 2):
	f = open_file(fname)

	S = []
	L = []
	max_len = 0
	idx = -1
	sentence_set = set()

	new_x = []
	new_y = []
	new_length = []
	for line in f:
		idx += 1

		if lower:
			line = line.lower()
		l = line.strip().split('\t')
		s = l[0].split(' ')
		label = int(l[1])

		if len(s) < min_len:
			continue
		for conj in conjunctions:
			if conj in l[0]:
				sentence_set.add(idx)

	for idx in sentence_set:
		if new_x == []:
			new_x = X_test[idx].reshape((1,X_test.shape[1]))
			new_y = Y_test[idx].reshape((1,5))
			new_length += [length_test[idx]]
			continue
		new_x = np.concatenate([new_x, X_test[idx].reshape((1,X_test.shape[1]))])
		new_y = np.concatenate([new_y, Y_test[idx].reshape((1,5))])
		new_length += [length_test[idx]]

	return new_x, new_y, new_length

def test_sa(model, X_test,Y_test, length_test):
	b_X_test, b_Y_test = distribute_buckets(length_test, [X_test], [Y_test], step_size = 1, x_set = set([0]), y_set = set())

	correct = 0.0
	count = 0.0

	polarity_cor = defaultdict(int)
	polarity_n = defaultdict(int)

	length_cor = defaultdict(int)
	length_n = defaultdict(int)

	for j,b in enumerate(b_X_test):
		[x_test] = b_X_test[j]
		[y_test] = b_Y_test[j]

		if len(x_test) == 0:
			continue
		prediction = model.predict(x_test, batch_size = 128, verbose = False)
		prediction_idx = np.argmax(prediction, axis = 1)
		gold_idx = np.argmax(y_test, axis = 1)

		analyze_length(length_cor, length_n, gold_idx, prediction_idx, [x_test.shape[1] - 1]* x_test.shape[0])
		analyze_polarity(polarity_cor, polarity_n, gold_idx, prediction_idx, [])

		c = np.count_nonzero(np.equal(prediction_idx,gold_idx))*1.00
		n = len(gold_idx)
#		print x_test.shape,  c / n
		correct += c
		count += n
	print '___'
	for key in length_cor:
		print key, length_cor[key] / length_n[key] 
	print '___'
	for key in polarity_cor:
		print key, polarity_cor[key] / polarity_n[key] 
	print '___'
	print count, correct, correct/count
	return {'length_cor' : length_cor, 'polarity_cor' : polarity_cor}

if __name__ == '__main__':

	prefix = sys.argv[1]
	fname = sys.argv[1] + '.analyze.txt'
	meta = prefix + '.meta'
	arch = prefix + '.arch'
	model_filename = prefix + '.model'

	meta_dict = pickle.load(open(meta))

	with open(arch) as json_file:
		architecture = json.load(json_file)
	model = model_from_json(architecture)
	model.load_weights(model_filename)

	X_test, Y_test, length_test = prepare_sa_test(meta_dict['dicts']['word_idx'])
#  	result = model.evaluate(X_test,Y_test, verbose = 0, batch_size=128, show_accuracy = True)
#   print result

 	print "______ all"
# 	print X_test.shape,Y_test.shape,len(length_test)
 	all_dict = test_sa(model,X_test,Y_test,length_test)

 	print "______ subset"
 	subset_x, subset_y, subset_length = get_s_conjunctions(X_test,Y_test,length_test)
 	subset_dict = test_sa(model, subset_x, subset_y, subset_length)

	print "_________________ conjunction"
	idx = 720
	x = X_test[idx][X_test.shape[1]-length_test[idx]:]
	x = x.reshape((1,length_test[idx]))

	conjunction = running_decode(model, meta_dict, x, Y_test[idx], fname = fname)

	print "_________________ negation"

	negation_orig = []
	negation_changes = []
	for sentence in ["roger dodger is one of the most compelling variations on this theme", "i liked every single minute of this film .", "it 's just incredibly dull ."]:

		instance = sentence.split(' ')  + ['</s>']
		x = np.array([[ meta_dict['dicts']['word_idx'][word]  for word in instance]], dtype = 'int64')
		y = np.array([[0,0,0,0,0]], dtype = np.int)

		negation_orig += [running_decode(model, meta_dict, x, y, fname = fname)]

	for sentence in ["roger dodger is one of the worst compelling variations on this theme", "i did n't like a single minute of this film .", "it 's definitely not dull ."]:

		instance = sentence.split(' ')  + ['</s>']
		x = np.array([[ meta_dict['dicts']['word_idx'][word]  for word in instance]], dtype = 'int64')
		y = np.array([[0,0,0,0,0]], dtype = np.int)

		negation_changes += [running_decode(model, meta_dict, x, y, fname = fname)]

	analyze = {'all_dict' : all_dict, 'subset_dict' : subset_dict, 'conjunction' : conjunction, 'negation_orig' : negation_orig, 'negation_changes' : negation_changes}
	pickle.dump(analyze, open(sys.argv[2],'w'))
