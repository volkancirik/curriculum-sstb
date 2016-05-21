import argparse

def get_sa():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch-size , default 64',type=int, default = 64)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 500',type=int, default = 500)

	parser.add_argument('--dropout', action='store', dest='dropout',help='dropout rate, default = 0.50', type=float, default = 0.50)

	parser.add_argument('--patience', action='store', dest='patience',help='# of epochs for patience, default = 20', type=int, default = 20)

	parser.add_argument('--prefix', action='store', dest='prefix',help='exp log prefix to append exp/{} default = DUMMY', default = 'DUMMY')

	parser.add_argument('--unit', action='store', dest='unit',help='train with {gru lstm base_soft pp kernel maxout} units,default = lstm', default = 'lstm')

	parser.add_argument('--hidden', action='store', dest='n_hidden',help='hidden size of softmax layer, default = 256', type=int, default = 256)

	parser.add_argument('--layers', action='store', dest='layers',help='# of RNN layers, default = 1', type=int, default = 1)

	parser.add_argument('--use-sentence', action='store_true', dest='sentence',help='use sentences(or flatten dataset), default : false')

	parser.add_argument('--seed', action='store', dest='seed',help='random seed, if seed = 0, do not randomize, default = 0',type = int, default = 0)

	parser.add_argument('--pretrained', action='store', dest='pretrained',help='{embeddings/{word2vec|glove}.pkl} use pretrained word embeddings default:"../embeddings/glove6B-100.pkl " ', default = '../embeddings/glove6B-100.pkl')

	parser.set_defaults(sentence = False)

	return parser


def get_sa_test():
	parser = argparse.ArgumentParser()

	parser.add_argument('--path', action='store', dest='path',help='path to .meta .arch and .model file', default = '')

	return parser

def get_embeddings(word_idx, idx_word, wvec = 'embeddings/word2vec.pkl', UNK_vmap = '*UNKNOWN*', expand_vocab = False, filtered = False):
	import gzip, sys
	import cPickle as pickle
	import numpy as np


	try:
		print >> sys.stderr, "loading word vectors..."
		v_map = pickle.load(gzip.open(wvec, "rb"))
		dim = len(list(v_map[UNK_vmap]))
		print >> sys.stderr, "%d dim word vectors are loaded.." % (dim)
	except:
		print >> sys.stderr, "word embedding file %s cannot be read." % (wvec)
		quit(1)

	print >> sys.stderr, "data vocab size %d vmap vocab size %d" % (len(word_idx),len(v_map))

	V_text = len(word_idx)

	if expand_vocab:
		for w in v_map.keys():
			if w not in word_idx:
				V_text += 1
				word_idx[w] = V_text
				idx_word[V_text] = w

	embedding_weights = np.zeros((V_text+1,dim))
	unk = 0
	n = 0
	for w in word_idx:
		n += 1
		idx = word_idx[w]
		if w not in v_map:
			w = UNK_vmap
			unk += 1
		try:
			embedding_weights[idx,:] = v_map[w]
		except:
			print >> sys.stderr, "something is wrong with following tuple:", idx, idx_word[idx], w
			quit(1)
	print >> sys.stderr, "%d/%d unknowns" % (unk,n)
	return embedding_weights, word_idx, idx_word
