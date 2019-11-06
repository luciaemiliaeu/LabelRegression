import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.decomposition import PCA
from matplotlib.patches import Polygon


datasets = [("./databases/iris.csv",3)]
#("./databases/mnist64.csv",10),("./databases/iris.csv",3),("./databases/vidros.csv",6), ("./databases/sementes.csv",3)]

def plotRegression(X, y, svr, attr):
	plt.figure()
	predicted = svr.predict(X)	
	
	'''
	pca = PCA(n_components=1)
	X_ = pca.fit(X).transform(X)
	plt.scatter(X_, y)
	plt.plot(X_, predicted)
	'''
	x = np.arange(X.shape[0])
	plt.plot(x, predicted, lw=2)
	plt.scatter(x, y, facecolor='none', edgecolor='b', s=50, label='support vectors')
	plt.scatter(x[np.setdiff1d(np.arange(len(x)), svr.support_)],y[np.setdiff1d(np.arange(len(x)), svr.support_)], facecolor="none", edgecolor="k", s=50, label='other training data')
	
	plt.legend()
	plt.title(attr)

def plotPrediction(attrName, predictions):
	num_cluster = np.unique(predictions.loc[:,'Cluster'].get_values()).shape[0]
	fig, axes = plt.subplots(nrows=num_cluster, ncols=1)
	fig.suptitle(attrName)

	# Dataframe : {y_real, y_Predicted, Cluster, Erro}
	for grupo, values in predictions.groupby(['Cluster']):
		values = values.sort_values(by=attr)
		values = values[['Actual', 'Predicted', 'Erro']]			
		values.plot(kind = 'bar', ax=axes[grupo-1], title=('Classe '+str(grupo)))

def plotResults(baseTitle, results):

	for attr, data in results.groupby(['Atributo']):
		num_cluster = np.unique(results.loc[:,'Cluster'].get_values()).shape[0]
		figE, axesE = plt.subplots(nrows=num_cluster, ncols =1)
		figE.suptitle(attr)

		for cluster, values in data.groupby(['Cluster']):
			attr_column = values.loc[values.index,'Saida'].get_values()
			erro = values.loc[:,'Erro'].get_values()

			axesE[cluster-1].plot(attr_column, erro, label='Erro real')
			
			poli = np.polyfit(attr_column.astype(float), erro.astype(float), 3)
			xx = np.linspace(min(attr_column), max(attr_column))
			yy = np.polyval(poli, xx)
			verts = [(min(attr_column),0), *zip(xx,yy), (max(attr_column),0)]
			poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
			
			axesE[cluster-1].add_patch(poly)
			axesE[cluster-1].plot(xx, yy , label='Apr. Poli.')
			axesE[cluster-1].legend()	



#['Cluster', 'Atributo', 'Saida', 'nor_Saida', 'Erro']
def erro_metrics(results):
	erroFuncs = pd.DataFrame(columns=['Cluster', 'Atributo', 'RMSE', 'AreaSobCurva'])
	
	for attr, data in results.groupby(['Atributo']):
		for cluster, values in data.groupby(['Cluster']):
			#rmse = sqrt(mean_squared_error(values.loc[:,'Predicted'], values.loc[:,'Actual']))

			attr_column = values.loc[values.index,'Saida'].get_values()
			erro = values.loc[:,'Erro'].get_values()
			poli = np.polyfit(attr_column.astype(float), erro.astype(float), 3)
			
			auc = integrate.trapz(np.poly1d(poli),min(attr_column), max(attr_column))
	
			print(auc)
			input()
			#erroFuncs.loc[erroFuncs.shape[0],:] = [cluster,attr, rmse, auc ]

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
		
		plotRegression(cluster.drop(attr, axis=1).get_values(),cluster.loc[:,attr].get_values(), svr_rbf, attr)

		# Dataframe : {y_real, y_Predicted, Cluster, Erro}
		y_ = pd.DataFrame({'Actual': y_test, 'Predicted': predicted, 'Cluster':  Y[test_set.index]})
		y_ = y_.assign(Erro=lambda x: abs(x.Actual-x.Predicted))
		y_.index = test_set.index
		y_ = y_.join(test_set.loc[y_.index, attr])
		
		#plotPrediction(attr, y_)

	
		# Separa os dados em grupos
		for c, data in y_.groupby(['Cluster']):			
			for out, values in data.groupby(attr):
				error.loc[error.shape[0],:] = [c, attr, X.loc[values.index[0],attr], out, values.mean(axis=0).Erro]	
	

	erro_metrics(error)
	#plotResults(title, error)

plt.show()
		