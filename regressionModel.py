import pandas as pd
import numpy as np
import statistics
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os

class trainingModels:
	def __init__(self, normalBD, attr_names, title, folds):
		_erros = pd.DataFrame(columns=['Atributo', 'mean_squared_error', 'r2'])
		_metrics = pd.DataFrame(columns=['Atributo', 'metric','mean','sd'])

		# predisctions: {'index', 'Atributo', 'predict'}
		self.predictions, erros = self.training(normalBD, attr_names, folds)
		
		#Cálculo das métricas de avaliação dos modelos de regressão
		for e in erros:
			_erros.loc[_erros.shape[0],:] = [e[0], e[1], e[2]]
		
		erro_ = [(attr, [x[1] for x in erros if x[0]==attr]) for attr in np.unique([x[0] for x in erros]).tolist() ]
		mean_error = [(attr, statistics.mean([x[1] for x in erros if x[0]==attr])) for attr in np.unique([x[0] for x in erros]).tolist()]
		sd_error = [(attr, statistics.stdev([x[1] for x in erros if x[0]==attr])) for attr in np.unique([x[0] for x in erros]).tolist()]
		for me, sd in zip(mean_error, sd_error):
			_metrics.loc[_metrics.shape[0],:] = [me[0], 'mean_squared_error', me[1], sd[1]]
		
		r2 = [(attr, [x[2] for x in erros if x[0]==attr]) for attr in np.unique([x[0] for x in erros]).tolist() ]
		mean_r2 = [(attr, statistics.mean([x[2] for x in erros if x[0]==attr])) for attr in np.unique([x[0] for x in erros]).tolist()]
		sd_r2 = [(attr, statistics.stdev([x[2] for x in erros if x[0]==attr])) for attr in np.unique([x[0] for x in erros]).tolist()]
		for me, sd in zip(mean_r2, sd_r2):
			_metrics.loc[_metrics.shape[0],:] = [me[0], 'r2', me[1], sd[1]]

		self.save_result(title, _erros, 'erroRegression.csv')
		self.save_result(title, _metrics, 'metricsRegression.csv')

	def save_result(self, dataset_name, table, file_name):
		script_dir = os.path.dirname(__file__)
		results_dir = os.path.join(script_dir, 'Testes/'+dataset_name+'/')
		sample_file_name = file_name

		if not os.path.isdir(results_dir):
		    os.makedirs(results_dir)
		table.to_csv(results_dir + file_name, index=False)
		

	def training(self, normalBD, attr_names, folds):
		# Cria DataFrames de treino e teste da bd normalizada 
		# y: atributo de saída
		kf = KFold(n_splits = folds, shuffle = True, random_state = 2)
		model = SVR(kernel='linear', C=100, gamma='auto')
		predict = pd.DataFrame(columns=['index', 'Atributo', 'predict'])
		erro_metrics = []
		for train, test in kf.split(normalBD):
			for attr in attr_names:		
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
		
		return predict , erro_metrics