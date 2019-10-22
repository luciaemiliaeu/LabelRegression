import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

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
	atributos_names = cluster.columns
	cluster = pd.DataFrame(MinMaxScaler().fit_transform(cluster.get_values()), columns = atributos_names)
	
	
	for attr in range(cluster.shape[1]):
		test_set = cluster.sample(frac=0.66)
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
		clt = classe[test_set.index]
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
			data = data.sort_values(by=atributos_names[attr])
			erro_relativo = data.loc[:,'Erro']
			attr_column = data.loc[:,atributos_names[attr]]
			
			end = np.shape(attr_column)[0]
			x = np.linspace(0, end, end, endpoint = True)
			interpolacao_linear = interp1d(x, erro_relativo, kind='linear')
			interpolacao_quadratic = interp1d(x, erro_relativo, kind='quadratic')
			interpolacao_cubic = interp1d(x, erro_relativo, kind='cubic')
			lin = interpolacao_linear(x)
			qua = interpolacao_quadratic(x)
			cub = interpolacao_cubic(x)


			axesE[c-1].plot(x, erro_relativo, x, lin, 'k--', legend)
			axesE[c-1].plot(x, erro_relativo, x, qua, 'k--')
			axesE[c-1].plot(x, erro_relativo, x, cub, 'k--')
			

			#x = np.arange(data.shape[0])
			#axesE[c-1].plot(x, erro_relativo)
			
			data = data[['Actual', 'Predicted', 'Erro']]			
			data.plot(kind = 'bar', ax=axes[c-1], title=('Classe '+str(c)))
	plt.show()
		

		