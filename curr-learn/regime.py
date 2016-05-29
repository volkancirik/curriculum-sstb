import random

def onepass_curr(data, model, EPOCH, RANDOMIZE, BATCH_SIZE, PATIENCE, NB, PREFIX, FOOTPRINT, ACCURACY):
	train_history = {'loss' : [], 'val_loss' : [], 'acc' : [], 'val_acc' : []}

	b_X_tr, b_Y_tr, X_val, Y_val = data
	pat = 0
	BLIST = range(NB)
	if ACCURACY:
		best_val = 0
	else:
		best_val = float('inf')
	criteron = 'val_acc' if ACCURACY else 'val_loss'

	for b_idx in BLIST:
		for iteration in xrange(EPOCH):
			train_history['loss'] += [0]

			[X_train] = b_X_tr[b_idx]
			[Y_train] = b_Y_tr[b_idx]

			if len(X_train) == 0:
				continue

			print('iteration {}/{} bucket {}/{}'.format(iteration+1,EPOCH, b_idx+1,NB))

			eh = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, nb_epoch = 1, verbose = False)
			for key in ['loss']:
				train_history[key][-1] += eh.history[key][0]

			vl = model.evaluate(X_val,Y_val, batch_size = BATCH_SIZE, verbose = False, show_accuracy = ACCURACY)
			train_history['val_loss'] += [vl]
			if ACCURACY:
				train_history['val_acc'] += [vl[1]]

			print("VAL {:10.4f} best VAL {:10.4f} no improvement in {}".format(train_history[criteron][-1],best_val,pat))

			if (train_history[criteron][-1] <= best_val and ACCURACY) or (train_history[criteron][-1] >= best_val and ACCURACY == False):
				pat += 1
			else:
				pat = 0
				best_val = train_history[criteron][-1]
				model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
			if pat == PATIENCE:
				break
		print('loading the previous best model')
		model.load_weights(PREFIX + FOOTPRINT + '.model')
		pat = 0

	return train_history

def curriculum(data, model, EPOCH, RANDOMIZE, BATCH_SIZE, PATIENCE, NB, PREFIX, FOOTPRINT, ACCURACY):
	train_history = {'loss' : [], 'val_loss' : [], 'acc' : [], 'val_acc' : []}

	b_X_tr, b_Y_tr, X_val, Y_val = data
	pat = 0
	BLIST = range(NB)
	if ACCURACY:
		best_val = 0
	else:
		best_val = float('inf')
	criteron = 'val_acc' if ACCURACY else 'val_loss'

	for b_idx in BLIST:
		for iteration in xrange(EPOCH):
			train_history['loss'] += [0]

			for j in xrange(b_idx+1):
				[X_train] = b_X_tr[j]
				[Y_train] = b_Y_tr[j]

				if len(X_train) == 0:
					continue

				print('iteration {}/{} bucket {}/{}'.format(iteration+1,EPOCH, j+1,NB))

				eh = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, nb_epoch = 1, verbose = False)
				for key in ['loss']:
					train_history[key][-1] += eh.history[key][0]

			vl = model.evaluate(X_val,Y_val, batch_size = BATCH_SIZE, verbose = False, show_accuracy = ACCURACY)
			train_history['val_loss'] += [vl]
			if ACCURACY:
				train_history['val_acc'] += [vl[1]]

			print("VAL {:10.4f} best VAL {:10.4f} no improvement in {}".format(train_history[criteron][-1],best_val,pat))

			if (train_history[criteron][-1] <= best_val and ACCURACY) or (train_history[criteron][-1] >= best_val and ACCURACY == False):
				pat += 1
			else:
				pat = 0
				best_val = train_history[criteron][-1]
				model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
			if pat == PATIENCE:
				break
		print('loading the previous best model')
		model.load_weights(PREFIX + FOOTPRINT + '.model')
		pat = 0

	return train_history


def vanilla(data, model, EPOCH, RANDOMIZE, BATCH_SIZE, PATIENCE, NB, PREFIX, FOOTPRINT, ACCURACY):
	train_history = {'loss' : [], 'val_loss' : [], 'acc' : [], 'val_acc' : []}

	b_X_tr, b_Y_tr, X_val, Y_val = data
	pat = 0
	BLIST = range(NB)
	if ACCURACY:
		best_val = 0
	else:
		best_val = float('inf')
	criteron = 'val_acc' if ACCURACY else 'val_loss'

	for iteration in xrange(EPOCH):
		train_history['loss'] += [0]

		if RANDOMIZE:
			random.shuffle(BLIST)

		for j in BLIST:
			[X_train] = b_X_tr[j]
			[Y_train] = b_Y_tr[j]

			if len(X_train) == 0:
				continue

			print('iteration {}/{} bucket {}/{}'.format(iteration+1,EPOCH, j+1,NB))

			eh = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, nb_epoch = 1, verbose = False)
			for key in ['loss']:
				train_history[key][-1] += eh.history[key][0]

		vl = model.evaluate(X_val,Y_val, batch_size = BATCH_SIZE, verbose = False, show_accuracy = ACCURACY)
		train_history['val_loss'] += [vl]
		if ACCURACY:
			train_history['val_acc'] += [vl[1]]

		print("VAL {:10.4f} best VAL {:10.4f} no improvement in {}".format(train_history[criteron][-1],best_val,pat))

		if (train_history[criteron][-1] <= best_val and ACCURACY) or (train_history[criteron][-1] >= best_val and ACCURACY == False):
			pat += 1
		else:
			pat = 0
			best_val = train_history[criteron][-1]
			model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
		if pat == PATIENCE:
			break
	return train_history
