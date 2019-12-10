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

	clusters = np.unique(predictions.loc[:,'Cluster'].to_numpy())
	fig, axes = plt.subplots(nrows=clusters.shape[0], ncols=1)
	fig.suptitle(attrName)
	ax = 0
	# Dataframe : {y_real, y_Predicted, Cluster, Erro}
	for i in clusters:
		values = predictions[(predictions['Cluster']==i)]
		values = values.sort_values(by='Actual')
		values = values[['Actual', 'Predicted', 'Erro']]			
		values.plot(kind = 'bar', ax=axes[ax], title=('Classe '+str(i)))
		ax += 1

def plotResults(baseTitle, results, polis, intersec):

	for attr, data in results.groupby(['Atributo']):
		plt.figure()
		plt.suptitle(attr)		
		for cluster, values in data.groupby(['Cluster']):

			attr_column = values.loc[values.index,'Saida'].to_numpy()
			erro = values.loc[:,'Erro'].to_numpy()
			
			poli = [p[0] for p in polis if p[1]==attr]
			pol = [p[0] for p in poli[0] if p[1]==cluster]

			if len(pol)>=1:
				min_ = min(attr_column)
				max_ = max(attr_column)
				xx = np.linspace(min_, max_)
				yy = np.polyval(pol[0], xx)
				verts = [(min(attr_column),0), *zip(xx,yy), (max(attr_column),0)]
				poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
				inter=[i[0] for i in intersec if i[1]==attr and i[2]==cluster]
				plt.scatter(inter, np.polyval(pol[0], inter))		
			
			plt.plot(xx, yy , label='Cluster' +str(cluster))
			plt.legend()

def plotPredictionMean(attrName, predictions):
	clusters = np.unique(predictions.loc[:,'Cluster'].to_numpy())
	fig, axes = plt.subplots(nrows=clusters.shape[0], ncols=1)
	fig.suptitle(attrName)
	ax = 0
	# Dataframe : {y_real, y_Predicted, Cluster, Erro}
	for i in clusters:
		values = predictions[(predictions['Cluster']==i)]
		values = values.sort_values(by='Saida')
		values[['Erro']].plot(kind='bar',ax=axes[ax], title=('Classe '+str(i)), xticks = values.Saida)
		ax += 1

'''def plotResults(baseTitle, results):

	for attr, data in results.groupby(['Atributo']):
		num_cluster = np.unique(results.loc[:,'Cluster'].to_numpy()).shape[0]
		figE, axesE = plt.subplots(nrows=num_cluster, ncols =1)
		figE.suptitle(attr)

		for cluster, values in data.groupby(['Cluster']):
			attr_column = values.loc[values.index,'Saida'].to_numpy()
			erro = values.loc[:,'Erro'].to_numpy()

			axesE[cluster-1].plot(attr_column, erro, label='Erro real')
			
			poli = np.polyfit(attr_column.astype(float), erro.astype(float), 3)
			xx = np.linspace(min(attr_column), max(attr_column))
			yy = np.polyval(poli, xx)
			verts = [(min(attr_column),0), *zip(xx,yy), (max(attr_column),0)]
			poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
			
			axesE[cluster-1].add_patch(poly)
			axesE[cluster-1].plot(xx, yy , label='Apr. Poli.')
			axesE[cluster-1].legend()			'''