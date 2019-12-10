import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plotRegression(X, y, svr, attr):
	plt.figure( figsize = [20,10])
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
	#plt.savefig('regression_'+attrName)

def plotPrediction(attrName, predictions):

	clusters = np.unique(predictions.loc[:,'Cluster'].to_numpy())
	fig, axes = plt.subplots(nrows=clusters.shape[0], ncols=1, figsize = [20,10])
	fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
	plt.xlabel('Element index')
	fig.suptitle(attrName)
	ax = 0
	# Dataframe : {y_real, y_Predicted, Cluster, Erro}
	for i in clusters:
		values = predictions[(predictions['Cluster']==i)]
		values = values.sort_values(by='Actual')
		values = values[['Actual', 'Predicted', 'Erro']]			
		values.plot(kind = 'bar', ax=axes[ax])
		axes[ax].set_title('Classe '+str(i),fontweight ='medium',  loc='left')
		ax += 1
	#plt.savefig('prediction_'+attrName)

def plotPredictionMean(attrName, predictions):
	clusters = np.unique(predictions.loc[:,'Cluster'].to_numpy())
	fig, axes = plt.subplots(nrows=clusters.shape[0], ncols=1, figsize = [20,10])
	fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
	plt.xlabel('Attr value')
	fig.suptitle(attrName)
	ax = 0
	# Dataframe : {y_real, y_Predicted, Cluster, Erro}
	for i in clusters:
		values = predictions[(predictions['Cluster']==i)]
		values = values.sort_values(by='Saida')
		values[['Erro']].plot(kind='bar',ax=axes[ax], xticks = values.Saida)
		axes[ax].set_title('Classe '+str(i),fontweight ='medium',  loc='left')
		ax += 1
	#plt.savefig('predictionMean_'+attrName)
'''
def plotResults(baseTitle, results, polis, intersec):

	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
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
		#plt.savefig('functions_'+attr)'''

def plotResults(baseTitle, results, polis, intersec, db):
	'Actual', 'Predicted', 'Atributo','Cluster', 'Erro'
	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
		plt.suptitle(attr)
		print(results.head())
		for cluster, values in data.groupby(['Cluster']):
			# Ploting dados 
			#points = db[(db['Atributo']==attr) & (db['Cluster']==cluster)][['Saida', 'Erro']]
			#p = plt.plot(points['Saida'], points['Erro'], 'o', label='Cluster '+str(cluster)+' data', alpha=0.3)
			
			
			p = plt.plot(values['Saida'], values['Erro'], 'o', markersize=7)
			color = p[0].get_color()
			# Ploting curvas
			attr_column = values.loc[values.index,'Saida'].values
			erro = values.loc[:,'Erro'].values
			
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
				plt.plot(inter, np.polyval(pol[0], inter), 'o', c='k')		
			plt.plot(xx, yy , label='Cluster' +str(cluster), c= color)
			
			plt.legend()
		
		#plt.savefig('functions_'+attr)