import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Polygon
import six

import saving_results as save

def render_outcomes_table(out, dataset_name, col_width=3.0, row_height=0.625, font_size=14,
					 header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
					 bbox=[0, 0, 1, 1], header_columns=0,
					 ax=None, **kwargs):
	# out: {'d', 'accuracys', 'n_elementosForLabel'}
	if ax is None:
		size = (np.array(out.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
		fig, ax = plt.subplots(figsize = size, dpi = 200)
		ax.axis('off')
	out.loc[:,['accuracys']]=results[['accuracys']].apply(lambda x: round(x.astype(np.double), 2), axis=1)
	mpl_table = ax.table(cellText=out.values, bbox=bbox, colLabels=out.columns, **kwargs)

	mpl_table.auto_set_font_size(False)
	mpl_table.set_fontsize(font_size)

	for k, cell in  six.iteritems(mpl_table._cells):
		cell.set_edgecolor(edge_color)
		if k[0] == 0 or k[1] < header_columns:
			cell.set_text_props(weight='bold', color='w')
			cell.set_facecolor(header_color)
		else:
			cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
	save.save_fig(dataset_name,'outcomes_'+dataset_name)
	plt.close('all')

def render_results_table(results, dataset_name, col_width=3.0, row_height=0.625, font_size=14,
					 header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
					 bbox=[0, 0, 1, 1], header_columns=0,
					 ax=None, **kwargs):
	# results: {'Cluster', 'Accuracy'}
	if ax is None:
		size = (np.array(results.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
		fig, ax = plt.subplots(figsize = size, dpi = 200)
		ax.axis('off')
	results.loc[:,['Accuracy']]=results[['Accuracy']].apply(lambda x: round(x.astype(np.double), 2), axis=1)
	mpl_table = ax.table(cellText=results.values, bbox=bbox, colLabels=results.columns, **kwargs)

	mpl_table.auto_set_font_size(False)
	mpl_table.set_fontsize(font_size)

	for k, cell in  six.iteritems(mpl_table._cells):
		cell.set_edgecolor(edge_color)
		if k[0] == 0 or k[1] < header_columns:
			cell.set_text_props(weight='bold', color='w')
			cell.set_facecolor(header_color)
		else:
			cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
	save.save_fig(dataset_name,'labels_accuracy_'+dataset_name)
	plt.close('all')

def render_labels_table(labels, dataset_name, col_width=3.0, row_height=0.625, font_size=14,
					 header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
					 bbox=[0, 0, 1, 1], header_columns=0,
					 ax=None, **kwargs):
	#label: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'Accuracy'}
	if ax is None:
		size = (np.array(labels.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
		fig, ax = plt.subplots(figsize = size, dpi = 200)
		ax.axis('off')
	labels.drop(['Accuracy'], axis=1, inplace=True)
	labels.loc[:,['min_faixa', 'max_faixa']]=labels[['min_faixa', 'max_faixa']].apply(lambda x: round(x.astype(np.double), 2), axis=1)
	mpl_table = ax.table(cellText=labels.values, bbox=bbox, colLabels=labels.columns, **kwargs)

	mpl_table.auto_set_font_size(False)
	mpl_table.set_fontsize(font_size)

	for k, cell in  six.iteritems(mpl_table._cells):
		cell.set_edgecolor(edge_color)
		if k[0] == 0 or k[1] < header_columns:
			cell.set_text_props(weight='bold', color='w')
			cell.set_facecolor(header_color)
		else:
			cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
	save.save_fig(dataset_name,'labels_'+dataset_name)
	plt.close('all')

def plot_Regression(X, y, svr, attr):
	plt.close('all')
	plt.figure(figsize = [20,10], dpi = 200)
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
	plt.close('all')

# Gráfico de barras com o valor real, as predições e o erro
def plot_Prediction(predictions, dataset_name):
	# predictions : {'Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'}
	clusters = predictions['Cluster'].unique()
	for attr, data in predictions.groupby(['Atributo']):
		fig, axes = plt.subplots(nrows=clusters.shape[0], ncols=1, figsize = [20,10], dpi = 200)
		fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
		plt.xlabel('Índice da instância')
		fig.suptitle(attr)
		ax = 0
		
		for i in clusters:
			values = data[(data['Cluster']==i)].sort_values(by='Actual')
			values = values[['Normalizado', 'Predicted', 'Erro']]			
			values.plot(kind = 'bar', ax=axes[ax])
			axes[ax].set_title('Grupo '+str(i), fontweight ='medium',  loc='left')
			ax += 1

		save.save_fig(dataset_name,'prediction_erro_'+attr)
		plt.close('all')

#Gráfico de barras com o erro médio de predição por ponto
def plot_Prediction_Mean_Erro(results, dataset_name):
	# results : {'Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'}
	clusters = results['Cluster'].unique()
	for attr, data in results.groupby(['Atributo']):
		fig, axes = plt.subplots(nrows=clusters.shape[0], ncols=1, figsize = [20,10], dpi = 200)
		fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
		plt.xlabel('Valor do atributo')
		fig.suptitle(attr)
		ax = 0
		
		for i in clusters:
			values = data[(data['Cluster']==i)].sort_values(by='Saida')
			values[['ErroMedio']].plot(kind='bar',ax=axes[ax], xticks = values.Saida)
			axes[ax].set_title('Grupo '+str(i),fontweight ='medium',  loc='left')
			ax += 1
		save.save_fig(dataset_name,'mean_prediction_erro_'+attr)
		plt.close('all')

# Gráfico de linhas com as funções e pontos de erro por grupo
def plot_Func_and_Points(results, polis, intersec, dataset_name):
	# results : {'Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'}
	# polis : [([funções], attr)]
	# intersec : [([pontos], atributo, grupo)]	
	for attr, data in results.groupby(['Atributo']):
		plt.figure(figsize = [20,10], dpi = 200)
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting erro 
			points = values[['Actual', 'Erro']]
			p = plt.plot(points['Actual'], points['Erro'], 'o', label='Cluster '+str(cluster)+' data', alpha=0.3)
			color = p[0].get_color()
			
			# Ploting curvas			
			attr_column = values['Actual'].values
			erro = values['Erro'].values
			
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
			plt.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster), c= color)
			plt.legend()
		
		save.save_fig(dataset_name,'functions_and_points_'+attr)
		plt.close('all')

# Gráfico de pontos dos erros médio por grupo
def plot_Mean_Points_Erro(results, dataset_name):
	# results : {'Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'}
	for attr, data in results.groupby(['Atributo']):
		plt.figure(figsize = [20,10], dpi = 200)
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting erro médio 
			points = values[['Saida', 'ErroMedio']]
			plt.plot(values['Saida'], values['ErroMedio'], 'o', markersize=7, label='Erros no Grupo '+str(cluster))			
			plt.legend()
		save.save_fig(dataset_name,'mean_Erro_Points_'+attr)
		plt.close('all')

# Gráfico de linha com a função do erro e pontos de erro médio por grupo
def plot_Func_and_PointsMean(results, polis, dataset_name):
	# results : {'Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'}
	# polis : [([funções], attr)]
	for attr, data in results.groupby(['Atributo']):
		plt.figure(figsize = [20,10], dpi = 200)
		plt.suptitle(attr)
		for cluster, values in data.groupby(['Cluster']):
			# Ploting erro médio 
			points = values[['Saida', 'ErroMedio']]
			p = plt.plot(values['Saida'], values['ErroMedio'], 'o', markersize=7, label='Erros no Grupo '+str(cluster))
			color = p[0].get_color()
			
			# Ploting curvas
			attr_column = values['Saida'].values
			erro = values['ErroMedio'].values
			
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
		save.save_fig(dataset_name,'func_pointsMean_'+attr)
		plt.close('all')

#Gáfico de linha das funções de erro por grupo
def plot_Functions(results, polis, dataset_name):
	# results : {'Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'}
	# polis : [([funções], attr)]
	for attr, data in results.groupby(['Atributo']):
		plt.figure(figsize = [20,10], dpi = 200)
		plt.suptitle(attr)
		p = []
		for cluster, values in data.groupby(['Cluster']):
			# Ploting funções
			attr_column = values['Saida'].values
			erro = values['ErroMedio'].values
			
			poli = [p[0] for p in polis if p[1]==attr]
			pol = [p[0] for p in poli[0] if p[1]==cluster]

			if len(pol)>=1:
				min_ = min(attr_column)
				max_ = max(attr_column)
				p.append(min_)
				p.append(max_)
				xx = np.linspace(min_, max_)
				yy = np.polyval(pol[0], xx)
		
				plt.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster))
			plt.legend()
		
		q = [ np.round(elem,2) for elem in p ]
		plt.xticks(list(set(q)))
		save.save_fig(dataset_name,'functions_'+attr)
		plt.close('all')

# Gráfico de linhas das funções de erro e as inteseções destacadas
def plot_Intersec(results, polis, intersec, dataset_name):
	# results : {'Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'}
	# polis : [([funções], attr)]
	# intersec : [([pontos], atributo, grupo)]	
	for attr, data in results.groupby(['Atributo']):
		plt.figure(figsize = [20,10], dpi = 200)
		plt.suptitle(attr)
		p =[]
		for cluster, values in data.groupby(['Cluster']):
			# Ploting curvas
			attr_column = values['Saida'].values
			erro = values['ErroMedio'].values
			
			poli = [p[0] for p in polis if p[1]==attr]
			pol = [p[0] for p in poli[0] if p[1]==cluster]

			if len(pol)>=1:
				min_ = min(attr_column)
				max_ = max(attr_column)
				xx = np.linspace(min_, max_)
				yy = np.polyval(pol[0], xx)

				inter=[i[0] for i in intersec if i[1]==attr and i[2]==cluster]
				for i in inter[0]: p.append(i)
				plt.plot(inter, np.polyval(pol[0], inter), 'o', c='k')				
				plt.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster))
			
			plt.legend()
		q = [  np.round(elem,2) for elem in p ]
		plt.xticks(list(set(q)))
		
		save.save_fig(dataset_name,'intersec_'+attr )
		plt.close('all')

# Gráfico de linhas das funções de erro e a AUC destacada
def plot_AUC(results, polis, labels, dataset_name):
	# results : {'Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'}
	# polis : [([funções], attr)]
	# labels: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'Accuracy'}	
	for attr, data in results.groupby(['Atributo']):
		fig, ax = plt.subplots(figsize = [20,10], dpi = 200)
		fig.suptitle(attr)
		
		for cluster, values in data.groupby(['Cluster']):
			l = labels[(labels['Atributo']== attr) & (labels['Cluster']== cluster)]
			# Ploting curvas
			attr_column = values['Saida'].values
			erro = values['ErroMedio'].values
			
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
		minimos = [x for x in labels[(labels['Atributo']== attr)].round(2)['min_faixa'].values]
		maximos = [x for x in labels[(labels['Atributo']== attr)].round(2)['max_faixa'].values]			
		ax.set_xticks(list(set(minimos+maximos)))
		ax.set_xticklabels(list(set(minimos+maximos)))
		save.save_fig(dataset_name,'AUC_'+attr )
		plt.close('all')
		
# Gráfio de linha das funções de erro e faixas limitadas
def plot_Limite_Points(results, polis, intersec, dataset_name):
	# results : {'Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'}
	# polis : [([funções], attr)]
	# intersec : [([pontos], atributo, grupo)]	
	erro_max = results['ErroMedio'].max()
	iy = np.linspace(0, erro_max)
	
	for attr, data in results.groupby(['Atributo']):
		fig, ax = plt.subplots(figsize = [20,10], dpi = 200)
		fig.suptitle(attr)
		
		for cluster, values in data.groupby(['Cluster']):
			# Ploting curvas
			attr_column = values.loc[values.index,'Saida'].values
			erro = values['ErroMedio'].values
			
			poli = [p[0] for p in polis if p[1]==attr]
			pol = [p[0] for p in poli[0] if p[1]==cluster]
 
			if len(pol)>=1:
				min_ = min(attr_column)
				max_ = max(attr_column)
				xx = np.linspace(min_, max_)
				yy = np.polyval(pol[0], xx)
				ax.plot(xx, yy , label='Apro. Poli. Grupo ' +str(cluster))
			
		
		points = [x[0] for x in intersec if x[1]==attr ][0]
		for i in points:	
			ix = [i]* iy.shape[0]
			ax.plot(ix, iy, '--', color= 'black')
		p = [  np.round(elem,2) for elem in points ]	
		ax.set_xticks(p)
		ax.set_xticklabels(p)
		
		save.save_fig(dataset_name,'limites_'+attr )
		plt.close('all')
