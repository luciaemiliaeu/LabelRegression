import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.patches import Polygon

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
		values = values.sort_values(by=attrName)
		values = values[['Actual', 'Predicted', 'Erro']]			
		values.plot(kind = 'bar', ax=axes[grupo-1], title=('Classe '+str(grupo)))

def plotResults(baseTitle, results, polis):
	cont = 0
	for attr, data in results.groupby(['Atributo']):
		plt.figure()
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			attr_column = values.loc[values.index,'Saida'].get_values()
			erro = values.loc[:,'Erro'].get_values()
		
			poli = polis[cont][cluster-1]
			xx = np.linspace(min(attr_column), max(attr_column))
			yy = np.polyval(poli, xx)
			verts = [(min(attr_column),0), *zip(xx,yy), (max(attr_column),0)]
			poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
			
			plt.plot(xx, yy , label='Cluster' +str(cluster))
			plt.legend()
		cont += 1
'''def plotResults(baseTitle, results):

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
			axesE[cluster-1].legend()			'''