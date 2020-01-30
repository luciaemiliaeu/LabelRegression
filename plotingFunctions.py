import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.patches import Polygon
import six

def render_mpl_table(set_name, data, col_width=3.0, row_height=0.625, font_size=14,
					 header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
					 bbox=[0, 0, 1, 1], header_columns=0,
					 ax=None, **kwargs):
	if ax is None:
		size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
		fig, ax = plt.subplots(figsize=size)
		ax.axis('off')
	data.drop('AUC', axis=1, inplace = True)
	data.loc[:,['min_faixa', 'max_faixa', 'Accuracy']]=data[['min_faixa', 'max_faixa', 'Accuracy']].apply(lambda x: round(x.astype(np.double), 2), axis=1)
	mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

	mpl_table.auto_set_font_size(False)
	mpl_table.set_fontsize(font_size)

	for k, cell in  six.iteritems(mpl_table._cells):
		cell.set_edgecolor(edge_color)
		if k[0] == 0 or k[1] < header_columns:
			cell.set_text_props(weight='bold', color='w')
			cell.set_facecolor(header_color)
		else:
			cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
	
	plt.savefig(set_name)

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
	plt.savefig('regression_'+attr)

def plotPrediction(attrName, predictions):

	clusters = np.unique(predictions.loc[:,'Cluster'].to_numpy())
	fig, axes = plt.subplots(nrows=clusters.shape[0], ncols=1, figsize = [20,10])
	fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
	plt.xlabel('Índice da instância')
	fig.suptitle(attrName)
	ax = 0
	# Dataframe : {y_real, y_Predicted, Cluster, Erro}
	for i in clusters:
		values = predictions[(predictions['Cluster']==i)]
		values = values.sort_values(by='Actual')
		values = values[['Actual', 'Predicted', 'Erro']]			
		values.plot(kind = 'bar', ax=axes[ax])
		axes[ax].set_title('Grupo '+str(i),fontweight ='medium',  loc='left')
		ax += 1
	plt.savefig('prediction_'+attrName)

def plotPredictionMean(attrName, predictions):
	clusters = np.unique(predictions.loc[:,'Cluster'].to_numpy())
	fig, axes = plt.subplots(nrows=clusters.shape[0], ncols=1, figsize = [20,10])
	fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
	plt.xlabel('Valor do atributo')
	fig.suptitle(attrName)
	ax = 0
	# Dataframe : {y_real, y_Predicted, Cluster, Erro}
	for i in clusters:
		values = predictions[(predictions['Cluster']==i)]
		values = values.sort_values(by='Saida')
		values[['Erro']].plot(kind='bar',ax=axes[ax], xticks = values.Saida)
		axes[ax].set_title('Grupo '+str(i),fontweight ='medium',  loc='left')
		ax += 1
	plt.savefig('predictionMean_'+attrName)

def plotResults(baseTitle, results, polis, intersec, db):
	
	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting dados 
			points = db[(db['Atributo']==attr) & (db['Cluster']==cluster)][['Saida', 'Erro']]
			p = plt.plot(points['Saida'], points['Erro'], 'o', label='Cluster '+str(cluster)+' data', alpha=0.3)
			
			
			#p = plt.plot(values['Saida'], values['Erro'], 'o', markersize=7, label='Erros no Grupo '+str(cluster))
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
				#plt.plot(inter, np.polyval(pol[0], inter), 'o', c='k')	
			plt.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster), c= color)
			
			plt.legend()
		
		#plt.savefig('functions_'+attr)

def plotPointsMean(results, db):
	
	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting dados 
			points = db[(db['Atributo']==attr) & (db['Cluster']==cluster)][['Saida', 'Erro']]
			p = plt.plot(values['Saida'], values['Erro'], 'o', markersize=7, label='Erros no Grupo '+str(cluster))			
			plt.legend()
		
		plt.savefig('meanPoints_'+attr)

def plotCurvesPointsMean(results, polis, db):
	
	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting dados 
			points = db[(db['Atributo']==attr) & (db['Cluster']==cluster)][['Saida', 'Erro']]
			p = plt.plot(values['Saida'], values['Erro'], 'o', markersize=7, label='Erros no Grupo '+str(cluster))
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

				plt.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster), c= color)
			
			plt.legend()
		
		plt.savefig('curve_meanPoints_'+attr)

def plotCurves(results, polis):
	
	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
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
		
				plt.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster))
			
			plt.legend()
		
		plt.savefig('curves_'+attr)

def plotIntersec(results, polis, intersec):
	
	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting curvas
			attr_column = values.loc[values.index,'Saida'].values
			erro = values.loc[:,'ErroMedio'].values
			
			poli = [p[0] for p in polis if p[1]==attr]
			pol = [p[0] for p in poli[0] if p[1]==cluster]

			if len(pol)>=1:
				min_ = min(attr_column)
				max_ = max(attr_column)
				xx = np.linspace(min_, max_)
				yy = np.polyval(pol[0], xx)

				inter=[i[0] for i in intersec if i[1]==attr and i[2]==cluster]
				plt.plot(inter, np.polyval(pol[0], inter), 'o', c='k')				
				plt.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster))
			
			plt.legend()
		
		plt.savefig('intersec_'+attr)

def plotAUC(results, polis, labels):
	
	for attr, data in results.groupby(['Atributo']):
		fig, ax = plt.subplots()
		fig.suptitle(attr)
		
		for cluster, values in data.groupby(['Cluster']):
			l = labels[(labels['Atributo']== attr) & (labels['Cluster']== cluster)]
			# Ploting curvas
			attr_column = values.loc[values.index,'Saida'].values
			erro = values.loc[:,'ErroMedio'].values
			
			poli = [p[0] for p in polis if p[1]==attr]
			pol = [p[0] for p in poli[0] if p[1]==cluster]

			if len(pol)>=1:
				min_ = min(attr_column)
				max_ = max(attr_column)
				xx = np.linspace(min_, max_)
				yy = np.polyval(pol[0], xx)
				c_ = ax.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster))
				color = c_[0].get_color()

				for i, row in l.iterrows():	
					a = row['min_faixa']
					b = row['max_faixa']
					ix = np.linspace(a,b)
					iy = np.polyval(pol[0], ix)
					v = [(a,0), *zip(ix,iy), (b,0)]
					p = Polygon(v, color=color, alpha=0.3)
					ax.add_patch(p)
			ax.legend()
		
		plt.savefig('AUC_'+attr)

def plotLimitePoints(results, polis, labels, db):
	
	for attr, data in results.groupby(['Atributo']):
		fig, ax = plt.subplots()
		fig.suptitle(attr)
		
		for cluster, values in data.groupby(['Cluster']):
			l = labels[(labels['Atributo']== attr) & (labels['Cluster']== cluster)]
			# Ploting dados 
			points = db[(db['Atributo']==attr) & (db['Cluster']==cluster)][['Saida', 'Erro']]
			p = plt.plot(points['Saida'], points['Erro'], 'o', label='Grupo '+str(cluster)+' data', alpha=0.3)
			color = p[0].get_color()

			# Ploting curvas
			attr_column = values.loc[values.index,'Saida'].values
			erro = values['Erro'].values
			
			poli = [p[0] for p in polis if p[1]==attr]
			pol = [p[0] for p in poli[0] if p[1]==cluster]

			if len(pol)>=1:
				min_ = min(attr_column)
				max_ = max(attr_column)
				xx = np.linspace(min_, max_)
				yy = np.polyval(pol[0], xx)
				ax.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster), color= color)

				erro_max = results['Erro'].max()
				iy = np.linspace(0, erro_max)
				
				for i, row in l.iterrows():	
					a = row['min_faixa']
					b = row['max_faixa']

					ix = [a]* iy.shape[0]
					ax.plot(ix, iy, '--', color= color)
					ix = [b]* iy.shape[0]
					ax.plot(ix, iy, '--', color= color)
			ax.legend()
		
		plt.savefig('Limit_'+attr)

def plotPoints(results, db):
	
	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting dados 
			points = db[(db['Atributo']==attr) & (db['Cluster']==cluster)][['Saida', 'Erro']]
			p = plt.plot(points['Saida'], points['Erro'], 'o', label='Grupo '+str(cluster)+' data', alpha=0.3)
			plt.legend()
		
		plt.savefig('points_'+attr)

def plotPointsCurve(results, polis, intersec, db):
	
	for attr, data in results.groupby(['Atributo']):
		plt.figure( figsize = [20,10])
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting dados 
			points = db[(db['Atributo']==attr) & (db['Cluster']==cluster)][['Saida', 'Erro']]
			p = plt.plot(points['Saida'], points['Erro'], 'o', label='Cluster '+str(cluster)+' data', alpha=0.3)
		
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
		
				inter=[i[0] for i in intersec if i[1]==attr and i[2]==cluster]				
				plt.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster), c= color)
			
			plt.legend()
		
		plt.savefig('curve_points_'+attr)

def plotData(db):
	grupos = db['classe'].unique()
	for i in grupos:
		x = db[(db['classe'] == i)]['X'].values
		y = db[(db['classe'] == i)]['Y'].values
		plt.plot(x, y, 'o', label = 'Grupo '+str(i))
	plt.legend()
	plt.savefig('data')

