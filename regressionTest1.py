import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

datasets = [("./databases/iris.csv",3)]
#,("./databases/vidros.csv",6), ("./databases/sementes.csv",3)]

for dataset, n_clusters in datasets:
	# Extrai o nome da base de dados
	title = dataset.split('/')[2].split('.')[0]+' dataset'
	print("")
	print(title)
	print("")	

	# Cria DataFrame com os valores de X e o cluster Y
	dataset = pd.read_csv(dataset, sep=',',parse_dates=True)
	classe = dataset.loc[:,'classe'].get_values()
	cluster = dataset.drop('classe', axis=1)
	atributos = cluster.columns

	plt.figure()
	for attr in range(cluster.shape[1]):
		train_set = cluster.sample(frac=0.33)		
		y_train = train_set.loc[:,cluster.columns[attr]].get_values()
		X_train = train_set.drop(cluster.columns[[attr]], axis=1).get_values()
		
		test_set = cluster.drop(train_set.index)
		y_test = train_set.loc[:,cluster.columns[attr]].get_values()
		X_test = train_set.drop(cluster.columns[[attr]], axis=1).get_values()

		# Treina o modelo de regressão 
		svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
		svr_rbf.fit(X_train,y_train)
		predicted = svr_rbf.fit(X_train,y_train).predict(X_test)
		

		for c in 
		# Ploting
		plt.subplot()
		y_ = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})
		y_.plot(kind='bar')
		plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
		plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
		plt.title(atributos[attr])

plt.show()
		