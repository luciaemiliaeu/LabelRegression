import pandas as pd
import numpy as np
import statistics
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


class trainingModels:
	def __init__(self, Y, normalBD, attr_names, n_exec):
	    # salava os retornos de cada execução: models, erros
		self.models = []
		r = []
		for i in range(n_exec):
			model, erros = self.training(normalBD, Y, attr_names,  0.33)
			r += erros
			self.models.append(model)
		self.erro_ = [(attr, [x[1] for x in r if x[0]==attr]) for attr in np.unique([x[0] for x in r]).tolist() ]
		self.mean_error = [(attr, statistics.mean([x[1] for x in r if x[0]==attr])) for attr in np.unique([x[0] for x in r]).tolist()]
		self.sd_error = [(attr, statistics.stdev([x[1] for x in r if x[0]==attr])) for attr in np.unique([x[0] for x in r]).tolist()]
		
		self.r2 = [(attr, [x[2] for x in r if x[0]==attr]) for attr in np.unique([x[0] for x in r]).tolist() ]
		self.mean_r2 = [(attr, statistics.mean([x[2] for x in r if x[0]==attr])) for attr in np.unique([x[0] for x in r]).tolist()]
		self.sd_r2 = [(attr, statistics.stdev([x[2] for x in r if x[0]==attr])) for attr in np.unique([x[0] for x in r]).tolist()]

	def training(self, normalBD, Y, attr_names,  pct):
		# Cria DataFrames de treino e teste da bd normalizada 
		# y: atributo de saída
		X_test, X_train, y_test, y_train = self.partitionDB(normalBD, Y, pct)

		# Treina os modelos, calcula r2 e RSME
		models = []
		erro_metrics = []
		for attr in attr_names:		
			x_test, x_train, attr_test, attr_train = self.partitionDBbyAttr(X_test, X_train, attr)
			model, y_Predicted = self.trainModel(x_train.to_numpy(), attr_train.to_numpy(), x_test.to_numpy())
			
			# Erros
			r2 = model.score(x_test, attr_test)
			erro = mean_squared_error(attr_test, y_Predicted)
			
			models.append((attr, model))
			erro_metrics.append((attr, erro, r2))
		return models, erro_metrics
	
	def partitionDB(self, X, Y, test_size):
		X_test = X.sample(frac=test_size)
		y_test = Y.loc[X_test.index]
		
		X_train = X.drop(X_test.index)
		y_train = Y.loc[X_train.index]	
		
		return X_test, X_train, y_test, y_train

	def partitionDBbyAttr(self, X_test, X_train, attr):
		attr_train = X_train.loc[:,attr]
		attr_test = X_test.loc[:,attr]

		x_train = X_train.drop(attr, axis=1)
		x_test = X_test.drop(attr, axis=1)
		
		return x_test, x_train, attr_test, attr_train

	def trainModel(self, X_train, y_train, X_test ):
		model = SVR(kernel='linear', C=100, gamma='auto').fit(X_train,y_train)
		y_predicted = model.predict(X_test)
		return model, y_predicted