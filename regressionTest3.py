import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

import matplotlib.pyplot as plt
from plotingFunctions import plotRegression, plotPrediction, plotResults
import scipy.integrate as integrate
from math import sqrt

datasets = [("./databases/iris.csv",3)]
#("./databases/mnist64.csv",10),("./databases/iris.csv",3),("./databases/vidros.csv",6), ("./databases/sementes.csv",3)]


def erro_metrics(results):
	erroFuncs = pd.DataFrame(columns=['Cluster', 'Atributo', 'RMSE', 'AreaSobCurva'])
	
	for attr, data in results.groupby(['Atributo']):
		for cluster, values in data.groupby(['Cluster']):
			
			attr_column = values.loc[values.index,'Saida'].get_values()
			erro = values.loc[:,'Erro'].get_values()
			poli = np.polyfit(attr_column.astype(float), erro.astype(float), 3)
			
			auc, err = integrate.quad(np.poly1d(poli),min(attr_column), max(attr_column))
			
			erroFuncs.loc[erroFuncs.shape[0],:] = [cluster, attr, rmse, auc ]

def polyApro(results):
	poli = []
	for attr, data in results.groupby(['Atributo']):
		poli.append(([np.polyfit(values.loc[values.index,'Saida'].get_values().astype(float), values.loc[:,'Erro'].get_values().astype(float), 3) for cluster, values in data.groupby(['Cluster'])], attr))
	return poli

def calErroFaixa(label, faixas, poli):
	#pra calcular o erro das faixas
	'''print(label)
	for i in faixas:
		print(i)
	for i in poli:
		print(i)
	'''

	for attr, data in label.groupby(['Atributo']):
		
		faixas_ = []
		for i in faixas:
			if i[1] == attr:
				faixas_ = i[0]
		for i in range(len(faixas_)-1):
			inicio = faixas_[i]
			fim = faixas_[i+1]
			clusters = []
			clusters = data[(data['minValue']<= inicio) & (data['maxValue']>=fim)]['Cluster'].get_values()
			


for dataset, n_clusters in datasets:
	# Extrai o nome da base de dados
	title = dataset.split('/')[2].split('.')[0]+' dataset'
	print("")
	print(title)
	print("")	

	# Cria DataFrame com os valores de X e o cluster Y
	dataset = pd.read_csv(dataset, sep=',',parse_dates=True)
	Y = dataset.loc[:,'classe'].get_values()
	X = dataset.drop('classe', axis=1)
	atributos_names = X.columns
	cluster = pd.DataFrame(X.apply(minmax_scale).get_values(), columns = atributos_names)
	
	error = pd.DataFrame(columns=['Cluster', 'Atributo', 'Saida', 'nor_Saida', 'Erro'])
	label = pd.DataFrame(columns=['Cluster', 'Atributo', 'minValue', 'maxValue', 'Erro'])
	for attr in cluster.columns:
		test_set = cluster.sample(frac=0.33)
		y_test = test_set.loc[:,attr].get_values()
		X_test = test_set.drop(attr, axis=1).get_values()

		train_set = cluster.drop(test_set.index)
		y_train = train_set.loc[:,attr].get_values()
		X_train = train_set.drop(attr, axis=1).get_values()
		

		# Treina o modelo de regressão 
		svr_rbf = SVR(kernel='linear', C=100, gamma='auto')
		svr_rbf.fit(X_train,y_train)
		predicted = svr_rbf.fit(X_train,y_train).predict(X_test)
		
		#plotRegression(cluster.drop(attr, axis=1).get_values(),cluster.loc[:,attr].get_values(), svr_rbf, attr)

		# Dataframe : {y_real, y_Predicted, Cluster, Erro}
		y_ = pd.DataFrame({'Actual': y_test, 'Predicted': predicted, 'Cluster':  Y[test_set.index]})
		y_ = y_.assign(Erro=lambda x: abs(x.Actual-x.Predicted))
		y_.index = test_set.index
		y_ = y_.join(test_set.loc[y_.index, attr])
		
		#plotPrediction(attr, y_)
		
		# Calcula o erro do attr sob a faixa toda
		for c, data in y_.groupby(['Cluster']):
			for out, values in data.groupby([attr]):
				error.loc[error.shape[0],:] = [c, attr, X.loc[values.index[0],attr], out, values.mean(axis=0).Erro]
			rsme = sqrt(mean_squared_error(data.loc[:,'Predicted'], data.loc[:,'Actual']))
			label.loc[label.shape[0],:] = [c, attr, X.loc[data.index, attr].min(), X.loc[data.index, attr].max(), rsme ]	
		#plotResults(title, error)
	polinomios = polyApro(error)
	#print(label.sort_values(by=['Atributo', 'Erro']))
	faixas = [((np.sort(np.unique(values[['minValue', 'maxValue']].get_values()))), out) for out, values in label.groupby(['Atributo'])]
	
	#plotResults(title, error, polinomios)
	calErroFaixa(label, faixas, polinomios)
	#erro_metrics(error)
	#print(label)
	
plt.show()
