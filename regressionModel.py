import pandas as pd
import numpy as np
import statistics
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import gc
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

class trainingModels:
	def __init__(self, normalBD, folds):
		'''
		self._erros = pd.DataFrame(columns=['Atributo', 'mean_squared_error', 'r2'])
		self._erros = self._erros.astype({'mean_squared_error': 'float64', 'r2': 'float64'})
		
		self._metrics = pd.DataFrame(columns=['Atributo', 'metric','mean','sd'])
		self._metrics = self._metrics.astype({'mean': 'float64', 'sd': 'float64'})
		'''
		# predisctions: {'index', 'Atributo', 'predict'}
		self.predictions, erros = self.training(normalBD, folds)
		'''
		#Cálculo das métricas de avaliação dos modelos de regressão
		for e in erros:
			self._erros.loc[self._erros.shape[0],:] = [e[0], e[1], e[2]]
		
		erro_ = [(attr, [x[1] for x in erros if x[0]==attr]) for attr in np.unique([x[0] for x in erros]).tolist() ]
		mean_error = [(attr, statistics.mean([x[1] for x in erros if x[0]==attr])) for attr in np.unique([x[0] for x in erros]).tolist()]
		sd_error = [(attr, statistics.stdev([x[1] for x in erros if x[0]==attr])) for attr in np.unique([x[0] for x in erros]).tolist()]
		for me, sd in zip(mean_error, sd_error):
			self._metrics.loc[self._metrics.shape[0],:] = [me[0], 'mean_squared_error', me[1], sd[1]]
		
		r2 = [(attr, [x[2] for x in erros if x[0]==attr]) for attr in np.unique([x[0] for x in erros]).tolist() ]
		mean_r2 = [(attr, statistics.mean([x[2] for x in erros if x[0]==attr])) for attr in np.unique([x[0] for x in erros]).tolist()]
		sd_r2 = [(attr, statistics.stdev([x[2] for x in erros if x[0]==attr])) for attr in np.unique([x[0] for x in erros]).tolist()]
		for me, sd in zip(mean_r2, sd_r2):
			self._metrics.loc[self._metrics.shape[0],:] = [me[0], 'r2', me[1], sd[1]]
		
		'''
	def training(self, normalBD, folds):
		# Cria DataFrames de treino e teste da bd normalizada 
		# y: atributo de saída
		print('Treinando ... ')
		
		predict = pd.DataFrame(columns=['index', 'Atributo', 'predict'])
		predict = predict.astype({'predict': 'float64'})
			
		erro_metrics = []
		n_attr = 1
		for attr in normalBD.columns:
			model = SVR(kernel='linear', C=100, gamma='auto')

			print(' attr ', n_attr)			
			Y = normalBD[attr]
			X = normalBD.drop(attr, axis=1)
			
			model.fit(X, Y)
			attr_predicted = model.predict(X)
			
			for i in X.index: 
				predict.loc[predict.shape[0],:] = [i, attr, attr_predicted[i]]

			r2 = model.score(X, Y)		
			erro = mean_squared_error(Y, attr_predicted)
			erro_metrics.append((attr, erro, r2))
			
			del model
			gc.collect()
			n_attr += 1
		return predict , erro_metrics
	'''

	def training(self, normalBD, folds):
		# Cria DataFrames de treino e teste da bd normalizada 
		# y: atributo de saída
		print('Treinando ... ')
		kf = KFold(n_splits = folds, shuffle = True, random_state = 2)
		model = SVR(kernel='linear', C=100, gamma='auto')
		
		predict = pd.DataFrame(columns=['index', 'Atributo', 'predict'])
		predict = predict.astype({'predict': 'float64'})

		n_folds = 1
		erro_metrics = []
		for train, test in kf.split(normalBD):
			n_attr = 1
			for attr in normalBD.columns:
				print('fold ', n_folds, ' attr ', n_attr)			
				attr_train = normalBD.loc[train,attr]
				attr_test = normalBD.loc[test,attr]

				x_train = normalBD.loc[train,:].drop(attr, axis=1)
				x_test = normalBD.loc[test,:].drop(attr, axis=1)

				model.fit(x_train, attr_train)
				attr_predicted = model.predict(x_test)

				for index, y in zip(test, attr_predicted):
					predict.loc[predict.shape[0],:] = [index, attr, y]

				r2 = model.score(x_test, attr_test)
				erro = mean_squared_error(attr_test, attr_predicted)
				erro_metrics.append((attr, erro, r2))
				
				n_attr += 1
			n_folds += 1
		return predict , erro_metrics
	'''