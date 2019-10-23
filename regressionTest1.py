import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error
from math import sqrt

datasets = [("./databases/iris.csv",3)]
#("./databases/mnist64.csv",10),("./databases/iris.csv",3),("./databases/vidros.csv",6), ("./databases/sementes.csv",3)]

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
		test_set = cluster.sample(frac=0.15)
		y_test = test_set.loc[:,cluster.columns[attr]].get_values()
		X_test = test_set.drop(cluster.columns[[attr]], axis=1).get_values()


		train_set = cluster.drop(test_set.index)
		y_train = train_set.loc[:,cluster.columns[attr]].get_values()
		X_train = train_set.drop(cluster.columns[[attr]], axis=1).get_values()
		

		# Treina o modelo de regressão 
		svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
		svr_rbf.fit(X_train,y_train)
		predicted = svr_rbf.fit(X_train,y_train).predict(X_test)
		
		# Dataframe : {y_real, y_Predicted, Classe, [attr]}
		clt = Y[test_set.index]
		y_ = pd.DataFrame({'Actual': y_test, 'Predicted': predicted, 'Cluster': clt})
		y_ = y_.assign(Erro=lambda x: abs(x.Actual-x.Predicted))
		y_.index = test_set.index
		y_ = y_.join(test_set)
		
		
		fig, axes = plt.subplots(nrows=3, ncols=1)
		figE, axesE = plt.subplots(nrows=3, ncols =1)
		
		fig.suptitle(atributos_names[attr])
		figE.suptitle(atributos_names[attr])

		
		# Separa os dados em grupos
		for c, data in y_.groupby(['Cluster']):
			rmse = sqrt(mean_squared_error(data.loc[:,'Predicted'], data.loc[:,'Actual']))
			print('RMSE attr',atributos_names[attr]," cluster",c,' :', rmse)
			error = pd.DataFrame(columns=['Cluster', 'Atributo', 'Saida', 'Erro'])
			for out, values in data.groupby(atributos_names[attr]):
				error.loc[error.shape[0],:] = [c,atributos_names[attr], out, values.mean(axis=0).Erro]
						
			q = X.loc[:, atributos_names[attr]]
			q = q[data.index]
			attr_column = np.unique(q.sort_values().get_values())

			
			a = input()


			axesE[c-1].plot(attr_column, error.loc[:, 'Erro'], label='Erro real')
			
			erro_relativo = error.loc[:,'Erro'].get_values()
			poli = np.polyfit(attr_column.astype(float), erro_relativo.astype(float), 3)

			xx = np.linspace(min(attr_column), max(attr_column))
			axesE[c-1].plot(xx, np.polyval(poli, xx), label='Apr. Poli.')
			axesE[c-1].legend()
			

			'''
			data = data.sort_values(by=atributos_names[attr])
			data = data[['Actual', 'Predicted', 'Erro']]			
			data.plot(kind = 'bar', ax=axes[c-1], title=('Classe '+str(c)))'''
			
	#plt.show()
		

		