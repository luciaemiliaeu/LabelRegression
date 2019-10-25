import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib.patches import Polygon
import scipy.integrate as integrate

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
	cluster = pd.DataFrame(MinMaxScaler().fit_transform(X.get_values()), columns = atributos_names)
	
	
	for attr in range(cluster.shape[1]):
		test_set = cluster.sample(frac=0.33)
		y_test = test_set.loc[:,cluster.columns[attr]].get_values()
		X_test = test_set.drop(cluster.columns[[attr]], axis=1).get_values()


		train_set = cluster.drop(test_set.index)
		y_train = train_set.loc[:,cluster.columns[attr]].get_values()
		X_train = train_set.drop(cluster.columns[[attr]], axis=1).get_values()
		

		# Treina o modelo de regressão 
		svr_rbf = SVR(kernel='linear', C=100, gamma='auto')
		svr_rbf.fit(X_train,y_train)
		predicted = svr_rbf.fit(X_train,y_train).predict(X_test)
		
		plotRegression(cluster.drop(cluster.columns[[attr]], axis=1).get_values(),cluster.loc[:,cluster.columns[attr]].get_values(), svr_rbf, atributos_names[attr])

		# Dataframe : {y_real, y_Predicted, Classe, [attr]}
		clt = Y[test_set.index]
		y_ = pd.DataFrame({'Actual': y_test, 'Predicted': predicted, 'Cluster': clt})
		y_ = y_.assign(Erro=lambda x: abs(x.Actual-x.Predicted))
		y_.index = test_set.index
		y_ = y_.join(test_set)
		


		print()
		fig, axes = plt.subplots(nrows=np.unique(Y).shape[0], ncols=1)
		figE, axesE = plt.subplots(nrows=np.unique(Y).shape[0], ncols =1)
		
		fig.suptitle(atributos_names[attr])
		figE.suptitle(atributos_names[attr])

		erroFuncs = pd.DataFrame(columns=['Cluster', 'Atributo', 'RMSE', 'AreaSobCurva'])
		# Separa os dados em grupos
		for c, data in y_.groupby(['Cluster']):
			error = pd.DataFrame(columns=['Cluster', 'Atributo', 'Saida', 'Erro'])
			for out, values in data.groupby(atributos_names[attr]):
				error.loc[error.shape[0],:] = [c,atributos_names[attr], out, values.mean(axis=0).Erro]
						
			q = X.loc[:, atributos_names[attr]]
			q = q[data.index]
			attr_column = np.unique(q.sort_values().get_values())

			
			#a = input()
			axesE[c-1].plot(attr_column, error.loc[:, 'Erro'], label='Erro real')
			
			erro_relativo = error.loc[:,'Erro'].get_values()
			poli = np.polyfit(attr_column.astype(float), erro_relativo.astype(float), 3)


			xx = np.linspace(min(attr_column), max(attr_column))
			yy = np.polyval(poli, xx)
			verts = [(min(attr_column),0), *zip(xx,yy), (max(attr_column),0)]
			poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
			

			axesE[c-1].add_patch(poly)
			axesE[c-1].plot(xx, yy , label='Apr. Poli.')
			axesE[c-1].legend()
						
			data = data.sort_values(by=atributos_names[attr])
			data = data[['Actual', 'Predicted', 'Erro']]			
			data.plot(kind = 'bar', ax=axes[c-1], title=('Classe '+str(c)))

			#auc = integrate.quad(yy,min(attr_column), max(attr_column))
			rmse = sqrt(mean_squared_error(data.loc[:,'Predicted'], data.loc[:,'Actual']))
			#erroFuncs.loc[erroFuncs.shape[0],:] = [c,atributos_names[attr], rmse, auc ]
			#print(erroFuncs)

	plt.show()
		

		