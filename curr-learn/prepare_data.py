# -*- coding: utf-8 -*-
#!/usr/bin/env/python
import sys, gzip
import cPickle as pickle
import numpy as np
from collections import defaultdict
from random import randint, shuffle
UNK='*UNKNOWN*'
THRESHOLD = 2
EOS = '</s>'


def sum_series(n, min_len = 2, max_len = 20, max_digit = 9, plus_sign = True, verbose = False, root = False):

	pairs = []
	seen = set()
	lengths = []
	if root:
		new_n = 0
		for j in xrange(min_len,max_len+1):
			new_n += min(10**j,n)

		len_tr = (new_n - n) + int(n * 0.6)
		len_val = int(n * 0.2)

		min_len = max_len
		lengths = [len_tr,len_val]
		n = new_n
	for j in xrange(min_len,max_len+1):
		nums = []

		i = 0
		if verbose:
			print >> sys.stderr, "starting length %d, there will be %d instances..." % (j,min(10**j,n))
		while i < min(10**j,n):
			serie = []
			for k in xrange(j):
				serie += [randint(0,max_digit)]

			str_serie = "".join([str(num) for num in serie])
			if str_serie in seen:
				continue

			i += 1
			seen.add(str_serie)
			nums.append(serie)

		shuffle(nums)
		pair_list = []
		for serie in nums:
			if plus_sign:
				x = "+".join([ str(num) for num in serie])
			else:
				x = [ str(num) for num in serie]

			pair_list += [(x, reduce(lambda a,b : a+b,serie))]
		pairs += [pair_list]
	extra_char = ''
	if plus_sign:
		extra_char = '+'
	return pairs, '0123456789' + extra_char, lengths

def shuffle_dataset(sequence, label):
	assert(len(sequence) == len(label))
	length = len(sequence)
	indexes = range(length)
	shuffle(indexes)
	return [sequence[idx] for idx in indexes],[label[idx] for idx in indexes]

def vectorize_sum_series(pairs, char_list, max_len = 40, lengths = []):

	tr_s = []
	tr_l = []
	val_s = []
	val_l = []
	test_s = []
	test_l = []
	char_list = [EOS,UNK] + list(char_list)

	for pl in pairs[:-1]:
		for i in xrange(len(pl)):
			x,y = pl[i]
			tr_s += [list(x)]
			tr_l += [y]

	pl = pairs[-1]
	if lengths != []:
		[len_tr, len_val] = lengths
	else:
		len_tr = int(len(pl) * 0.6)
		len_val = int(len(pl) * 0.2)

	for i in xrange(len_tr):
		x,y = pl[i]
		tr_s += [list(x)]
		tr_l += [y]
	for i in xrange(len_tr,len_tr+len_val):
		x,y = pl[i]
		val_s += [list(x)]
		val_l += [y]
	for i in xrange(len_tr+len_val,len(pl)):
		x,y = pl[i]
		test_s += [list(x)]
		test_l += [y]

	tr_s, tr_l = shuffle_dataset(tr_s,tr_l)
	val_s, val_l = shuffle_dataset(val_s,val_l)
	test_s, test_l = shuffle_dataset(test_s,test_l)

	word_idx = dict((c, i+1) for i, c in enumerate(char_list))
	idx_word = dict((i+1, c) for i, c in enumerate(char_list))

	X_tr, Y_tr, length_tr = vectorize_flat(tr_s, tr_l, word_idx, max_len, regression = True, eos = False)
	X_val, Y_val, length_val = vectorize_flat(val_s, val_l, word_idx, max_len, regression = True, eos = False)
	X_test, Y_test, length_test = vectorize_flat(test_s, test_l, word_idx, max_len, regression = True, eos = False)

	index = 0

	return X_tr, Y_tr, X_val, Y_val, X_test, Y_test, {'word_idx' : word_idx, 'idx_word' : idx_word} , [length_tr,length_val,length_test]

def prepare_ss(args = {'n' : 100, 'min_len' : 2, 'max_len' : 20, 'plus_sign' : True}, root = None):
	n = args['n']
	min_len = args['min_len']
	max_len = args['max_len']
	plus_sign = args['plus_sign']

	pairs, char_list, lengths = sum_series(n, min_len = min_len, max_len = max_len, plus_sign = plus_sign, root = 'root' == root)
	return vectorize_sum_series(pairs, char_list, max_len * 2  if plus_sign else max_len, lengths = lengths)

def prepare_ss_test(dicts, args = {'n' : 100, 'min_len' : 2, 'max_len' : 20, 'plus_sign' : True}):
	n = args['n']
	min_len = args['min_len']
	max_len = args['max_len']
	plus_sign = args['plus_sign']

	pairs, _ , _ = sum_series(n, min_len = max_len, max_len = max_len, plus_sign = plus_sign, root = 'root')
	test_s = []
	test_l = []
	for pl in pairs:
		for i in xrange(len(pl)):
			x,y = pl[i]
			test_s += [list(x)]
			test_l += [y]

	X_test, Y_test, length_test = vectorize_flat(test_s, test_l, dicts['word_idx'], max_len, regression = True, eos = False)
	return X_test, Y_test

def open_file(fname):
	try:
		f = open(fname)
	except:
		print >> sys.stderr, "file %s could not be opened" % (fname)
		quit(0)
	return f

def vectorize(data, word_idx, max_len, embedding = True):

	N = len(data)
	Y = np.zeros((N,5))
	if embedding:
		X = np.zeros((N,max_len), dtype = 'int64')

	length = []
	for i,ins in enumerate(data):
		pad = max_len - len(ins['s']) - 1 # for EOS
		for j,tok in enumerate(ins['s']+[EOS]):
			idx = word_idx.get(tok, word_idx[UNK])
			if embedding:
				X[i,pad + j] = idx
		Y[i, int((ins['l'] - 1e-10) / 0.2)] = 1
		length += [len(ins['s']) + 1]
	return X,Y, length


def get_flat(fname, lower = True, min_len = 2):
	f = open_file(fname)

	S = []
	L = []
	max_len = 0
	for line in f:
		if lower:
			line = line.lower()
		l = line.strip().split('\t')
		s = l[0].split(' ')
		label = int(l[1])

		if len(s) < min_len:
			continue

		S += [s]
		L += [label]
		max_len = max_len if max_len >= len(S[-1]) else len(S[-1])

	return S, L, max_len+1

def get_vocabs(data):

	vocab = defaultdict(int)

	for dsplit in data:
		for s in dsplit:
			for token in s:
				vocab[token] += 1

	words = [w for w in vocab]
	for w in words:
		if vocab[w] < THRESHOLD:
			del vocab[w]

	vocab[UNK] = THRESHOLD
	vocab[EOS] = THRESHOLD
	word_idx = dict((c, i) for i, c in enumerate(vocab))
	idx_word = dict((i, c) for i, c in enumerate(vocab))
	V = len(word_idx)
	first_w = idx_word[0]
	idx_word[0] = '*dummy*'
	idx_word[V] = first_w
	word_idx[first_w] = V
	word_idx['*dummy*'] = 0
	return word_idx, idx_word

def vectorize_flat(sentences,labels, word_idx, max_len, regression = False, eos = True):

#	print >> sys.stderr, "EOS is",eos
	N = len(sentences)
	if regression:
		Y = np.zeros((N), dtype=np.int)
	else:
		Y = np.zeros((N,5))
	X = np.zeros((N,max_len), dtype = 'int64')

	length = []
	for i,(s,l) in enumerate(zip(sentences,labels)):
		if eos:
			pad = max_len - len(s) - 1 # for EOS
			seq = s+[EOS]
		else:
			pad = max_len - len(s)
			seq = s
		for j,tok in enumerate(seq):
			idx = word_idx.get(tok, word_idx[UNK])
			X[i,pad + j] = idx
		if regression:
			Y[i] = l
		else:
			Y[i, l] = 1
		length += [len(seq)]
	return X,Y, length

def prepare_sa(args = { 'prefix' : '../data/', 'train_f' : 'train_', 'val_f' : 'dev_root.txt', 'test_f' : 'test_root.txt'}, root = 'all' ):
	prefix = args['prefix']
	train_f = args['train_f'] + root + '.txt'
	val_f = args['val_f']
	test_f = args['test_f']

	tr_s, tr_l, max_len_tr = get_flat(prefix + train_f)
	val_s,val_l, max_len_val = get_flat(prefix + val_f)
	test_s, test_l, max_len_test = get_flat(prefix + test_f)
	word_idx, idx_word = get_vocabs([tr_s,val_s])

	X_tr, Y_tr, length_tr = vectorize_flat(tr_s, tr_l, word_idx, max_len_tr)
	X_val, Y_val, length_val = vectorize_flat(val_s, val_l, word_idx, max_len_val)
	X_test, Y_test, length_test = vectorize_flat(test_s, test_l, word_idx, max_len_test)

	return X_tr, Y_tr, X_val, Y_val, X_test, Y_test, {'word_idx' : word_idx, 'idx_word' : idx_word} , [length_tr,length_val,length_test]

def prepare_sa_test(word_idx,prefix = '../data/', test_f = 'test_root.txt'):

	test_s, test_l, max_len_test = get_flat(prefix + test_f)
	X_test, Y_test, length_test = vectorize_flat(test_s, test_l, word_idx, max_len_test, eos = True)

	return X_test, Y_test, length_test

def prepare_data(dataset_id, args , root = 'all', clip = 1):
	datasets = { 'sa' : prepare_sa, 'ss' : prepare_ss}
	X_tr, Y_tr, X_val, Y_val, X_test, Y_test, dicts , [length_tr,length_val,length_test] = datasets[dataset_id](args = args, root = root)
	n_tr = int(len(X_tr) * clip)

	return X_tr[:n_tr], Y_tr[:n_tr], X_val, Y_val, X_test, Y_test, dicts , [length_tr[:n_tr],length_val,length_test]

def prepare_test(dataset_id, dicts, args , root = 'all', clip = 1):
	datasets = { 'sa' : prepare_sa_test, 'ss' : prepare_ss_test}
	X_test, Y_test = datasets[dataset_id](dicts, args = args)
	return X_test, Y_test

if __name__ == '__main__':
	from buckets import distribute_buckets
	dataset = 'sa'
	root = 'all'
	clip = 1
	pd_args = { 'sa' : { 'prefix' : '../data/', 'train_f' : 'train_', 'val_f' : 'dev_root.txt', 'test_f' : 'test_root.txt'}, 'ss' : {'n' : 100, 'min_len' : 2, 'max_len' : 20, 'plus_sign' : False} }

	X_tr, Y_tr, X_val, Y_val, X_test, Y_test, dicts, [length_tr,_,_] = prepare_data(dataset, pd_args[dataset] , root = root, clip = clip)
	print(X_tr.shape, Y_tr.shape)
	b_X_tr, b_Y_tr = distribute_buckets(length_tr, [X_tr], [Y_tr], step_size = 1, x_set = set([0]), y_set = set())
	for b in b_X_tr:
		print b[0].shape
