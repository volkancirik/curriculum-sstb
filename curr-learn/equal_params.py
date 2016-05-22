from prepare_data import prepare_data
from utils import get_sa, get_embeddings
from get_model import get_model

parser = get_sa()
p = parser.parse_args()

EPOCH = p.n_epochs
BATCH_SIZE = p.batch_size

pd_args = { 'sa' : { 'prefix' : '../data/', 'train_f' : 'train_', 'val_f' : 'dev_root.txt', 'test_f' : 'test_root.txt'}, 'ss' : {'n' : 100, 'max_len' : 20, 'plus_sign' : True} }
X_tr, Y_tr, X_val, Y_val, X_test, Y_test, dicts, [length_tr,_,_] = prepare_data(p.dataset, pd_args[p.dataset] , clip = 1)

embedding_weights = None
if p.pretrained != '':
	embedding_weights,word_idx, idx_word = get_embeddings(dicts['word_idx'],dicts['idx_word'], wvec = p.pretrained)
	dicts['word_idx'] = word_idx
	dicts['idx_word'] = idx_word
	FIXED = embedding_weights.shape[0] * embedding_weights.shape[1]
else:
	print("learning embeddings from data")

model, PREFIX, FOOTPRINT, RANDOMIZE = get_model(p, dicts, embedding_weights = embedding_weights)
BASE_N_PARAMETER = model.get_n_params()
print "%d parameters for model: --unit %s --hidden %d --layers %d" % (BASE_N_PARAMETER - FIXED, p.unit, p.n_hidden, p.layers)
prec = 0.0002
for unit in ['gru','bigru']:
 	p.unit = unit
 	model, PREFIX, FOOTPRINT, RANDOMIZE = get_model(p, dicts, embedding_weights = embedding_weights)

	h_min = 0
	h_max = 1024
	HIDDEN_SIZE = 0
	p.n_hidden  = HIDDEN_SIZE
	m_param = 0

	while not (BASE_N_PARAMETER - BASE_N_PARAMETER*prec <= m_param <=  BASE_N_PARAMETER + BASE_N_PARAMETER*prec):
		if m_param >= BASE_N_PARAMETER:
			h_max = HIDDEN_SIZE
		else:
			h_min = HIDDEN_SIZE
		HIDDEN_SIZE = (h_min + h_max)/2
		p.n_hidden  = HIDDEN_SIZE
		model, PREFIX, FOOTPRINT, RANDOMIZE = get_model(p, dicts, embedding_weights = embedding_weights)
		m_param = model.get_n_params()
		print "-->", unit, m_param, HIDDEN_SIZE

	print "%d parameters for model: --unit %s --hidden %d --layers %d" % (m_param - FIXED,p.unit,p.n_hidden, p.layers)
