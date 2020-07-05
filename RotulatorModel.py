import numpy as np
import pandas as pd
import scipy.integrate as integrate
from itertools import combinations
from scipy.optimize import fsolve
from sklearn.preprocessing import minmax_scale
from collections import defaultdict

import plotingFunctions as pltFunc
import savingResults as save
from regressionModel import trainingModels
from rotulate import calLabel

import random as rand


class Rotulator:
	def __init__(self, dataset, d, t, folds, title):		
		# DataFrames
		# X: atributos, Y: cluster, normalAttr: X normalizado
		# attr_names : lista de nomes dos atributos
		self.db, X, Y, normalAttr, attr_names = self.importBD(dataset)

		# Constrói os modelos de regressão e retorna um dataframe com as predições
		# predisctions: {'index', 'Atributo', 'predict'}
		models = trainingModels( normalAttr, folds)
		predictions = models.predictions
		print("regressions done")
		
		#Estrutura de dados para armazenar o erro das predições
		yy = pd.DataFrame(columns= ['Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'])
		
		for attr in attr_names:
			#seleciona as predições para o atributo attr
			y = predictions[(predictions['Atributo']==attr)].sort_values(by='index')
			y.index = y['index'].values
			y = y['predict']
			
			y_ = pd.DataFrame(columns=['Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'])
			y_['Actual'] = X[attr]
			y_['Normalizado'] = normalAttr[attr].values
			y_['Predicted'] = y.values
			y_['Cluster'] = Y.values
			y_ = y_.assign(Erro=lambda x: abs(x.Normalizado-x.Predicted))	
			y_ = y_.assign(Atributo=attr)

			yy = pd.concat([yy, y_])
		
		
		# Estrutura de dados pra armazenar o erro médio em cada ponto único do atributo por grupo
		errorByValue = pd.DataFrame(columns=['Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'])
		errorByValue = errorByValue.astype({'Saida': 'float64', 'nor_Saida': 'float64', 'ErroMedio': 'float64'})
		
		# Estrutura de dados pra armazenar o início e fim dos atributos em cada grupo
		attrRangeByGroup = pd.DataFrame(columns=['Cluster', 'Atributo', 'minValue', 'maxValue'])
		attrRangeByGroup = attrRangeByGroup.astype({'minValue': 'float64', 'maxValue': 'float64'})
		
		yy = yy.astype({'Actual': 'float64', 'Normalizado': 'float64', 'Predicted': 'float64', 'Erro':'float64'})
		for atributo, info in yy.groupby(['Atributo']):
			for clt, data in info.groupby(['Cluster']):
				# Calcula o mínimo e máximo do atributo no grupo
				attrRangeByGroup.loc[attrRangeByGroup.shape[0],:] = [clt, atributo, data['Actual'].min(), data['Actual'].max()]		
				# Calcula o erro médio em cada ponto único do atributo por grupo
				for out, values in data.groupby(['Actual']):
					errorByValue.loc[errorByValue.shape[0],:] = [clt, atributo, out, values.mean(axis=0).Normalizado, values.mean(axis=0).Erro]
		print('matrix done')
		
		# polynomials : funções polinomiais dos atributos por grupo
		# polynomials = [([([coef], grupo)], attr)]
		polynomials = self.polyApro(errorByValue)
		print('polynomials done')

		# calcula a relevância dos intervalos
		# rangeAUC: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'}
		rangeAUC = self.calAUCRange(attrRangeByGroup, polynomials, d)
		print("range done")
		
		# monta os rótulos
		# results: {'Cluster', 'Accuracy'}
		# label: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'Accuracy'}
		ranged_attr, self.results, self.labels, rotulation_process = calLabel(rangeAUC, t, self.db)
		print("rotulation done")
		
		save.save_table(title, ranged_attr, 'atributos_ordenados_por_acerto.csv')
		save.save_table(title, models._erros, 'erroRegression.csv')
		save.save_table(title, models._metrics, 'metricsRegression.csv')
		save.save_table(title, yy, 'predictions.csv')
		save.save_table(title, rangeAUC, 'range.csv')
		save.save_table(title, self.results, 'acuracia.csv')
		save.save_table(title, self.labels, 'rotulos.csv')
		save.save_table(title, rotulation_process, 'rotulos_por_iteracao.csv')
	
		pltFunc.plot_Prediction(yy,title)
		pltFunc.plot_Prediction_Mean_Erro(errorByValue, title)
		pltFunc.plot_Func_and_Points(yy, polynomials,  title)
		pltFunc.plot_Mean_Points_Erro(errorByValue, title)
		pltFunc.plot_Func_and_PointsMean(errorByValue, polynomials, title)
		pltFunc.plot_Functions(errorByValue, polynomials, title)
		#pltFunc.plot_Intersec(errorByValue, polynomials, intersecByAttrInCluster, title)
		pltFunc.plot_AUC(errorByValue, polynomials, rangeAUC, title)
		#pltFunc.plot_Limite_Points(errorByValue, polynomials, limitPoints, title)
		
		pltFunc.render_results_table(self.results, title, header_columns=0, col_width=2.0)
		pltFunc.render_labels_table( self.labels, title, header_columns=0, col_width=2.0)
		
	def importBD(self, path):
		#Carrega a base de dados e separa os atributos(X) do grupo(Y).
		dataset = pd.read_csv(path, sep=',',parse_dates=True)
		Y = dataset.loc[:,'classe']
		X = dataset.drop('classe', axis=1)
		
		#Normaliza os atributos
		normalAttr = pd.DataFrame(X.apply(minmax_scale).values, columns = X.columns)
		normalAttr = normalAttr.astype('float64')

		#Retorna a base de dados original, os atributos(X), os grupos(Y),
		# os atributos normalizados (normalAttr) e a lista de atributos (attr_names)
		return dataset, X, Y, normalAttr, X.columns
	
	def polyApro(self, results):
		polynomials = {}
		for attr, values in results.groupby(['Atributo']):
			d = {}
			for clt , data in values.groupby(['Cluster']):
				if data.shape[0]>1:
					d[clt] = np.polyfit(data['Saida'].to_numpy().astype(float), data['ErroMedio'].to_numpy().astype(float), 2)
					polynomials[attr] = d

		return polynomials
	
	def intersections(self, label, polynomials):
		intersections_points_by_attr = {}
		intersections_points_and_clusters_by_attr = {}
		
		for attr, values in label.groupby('Atributo'): 
			d = {}
			inter = []

			clusters = list(combinations(label['Cluster'].unique(),2))
			for c in clusters:
				x_max = values[(values['Cluster']==c[0]) | (values['Cluster']==c[1])].min()['maxValue']
				x_min = values[(values['Cluster']==c[0]) | (values['Cluster']==c[1])].max()['minValue']
				if x_min<x_max:
					# divide o intervalo em 5 partes iguais
					xx = np.linspace(x_min,x_max, num=5)

					#seleciona os polinômios dos grupos c[0] e c[1]
					poly1 = polynomials[attr][c[0]]
					poly2 = polynomials[attr][c[1]]
					
					# calcula as interseções em cada intervalo
					for x0 in xx:
						r = fsolve(lambda x : np.polyval(poly1, x) - np.polyval(poly2, x),x0, full_output=True, factor = 10) 
						if (r[3] == 'The solution converged.' and r[0][0] >= x_min and r[0][0]<= x_max):
							d[round(r[0][0],2)] = c
							inter.append(round(r[0][0],2))
			
			intersections_points_by_attr[attr] = list(set(inter))
			intersections_points_and_clusters_by_attr[attr] = d
		
		#for i in list(intersections_dict): print(i, intersections_dict[i])
		#for i in list(intersections): print(i, intersections[i])
		return intersections_points_by_attr, intersections_points_and_clusters_by_attr

	def get_groups(self, a, b, data, d, poli, erroFaixa, attr, sentido, extendendo):
		continuar = False

		inicio = a
		fim = b

		# seleciona os grupos com domínio da função do erro na faixa
		clusters = data[(data['minValue']<= inicio) & (data['maxValue']>=fim)]['Cluster'].values
		
		if clusters.shape[0]>=1:
			# calcula o erro para cada cluster
			errors = []
			for k in clusters:
				errors.append((k, self.AUC(inicio, fim, poli[k])))
			errors.sort(key=lambda tup: tup[1], reverse=False)
			
			# seleciona os grupos para os quais a faixa será atribuída com base no parâmetro d
			eminimo = errors[0][1]
			finalClusters = [i[0] for i in errors if i[1] <= (eminimo + (d*eminimo))]
			
			# Verifica se é necessário concatenar
			if sentido == 'front':
				for (clt, auc) in [i for i in errors if i[0] in finalClusters]:
					if not erroFaixa[(erroFaixa['max_faixa'] == inicio) & (erroFaixa['Cluster'] == clt)].empty:
						if erroFaixa[(erroFaixa['min_faixa'] >= inicio ) & (erroFaixa['max_faixa'] <= fim ) & (erroFaixa['Cluster'] == clt)].empty: 
							erroFaixa.loc[erroFaixa[(erroFaixa['Atributo'] == attr) & (erroFaixa['max_faixa'] == inicio) & (erroFaixa['Cluster'] == clt)].index, 'AUC'] += auc
							erroFaixa.loc[erroFaixa[(erroFaixa['Atributo'] == attr) & (erroFaixa['max_faixa'] == inicio) & (erroFaixa['Cluster'] == clt)].index, 'max_faixa'] = fim
							if extendendo: 
								continuar = True 
					else: 
						if not extendendo:
							erroFaixa.loc[erroFaixa.shape[0],:] = [clt, attr, inicio, fim, auc]
			
			if sentido == 'back':
				for (clt, auc) in [i for i in errors if i[0] in finalClusters]:
					# se existe um range na tabela que começa no fim do intervalo
					if not erroFaixa[(erroFaixa['min_faixa'] == fim) & (erroFaixa['Cluster'] == clt)].empty:
						# e se o intervalo não está contido em outro range
						if erroFaixa[(erroFaixa['min_faixa'] >= inicio ) & (erroFaixa['max_faixa'] <= fim ) & (erroFaixa['Cluster'] == clt)].empty: 
							# extende o range
							erroFaixa.loc[erroFaixa[(erroFaixa['Atributo'] == attr) & (erroFaixa['min_faixa'] == fim) & (erroFaixa['Cluster'] == clt)].index, 'AUC'] += auc
							erroFaixa.loc[erroFaixa[(erroFaixa['Atributo'] == attr) & (erroFaixa['min_faixa'] == fim) & (erroFaixa['Cluster'] == clt)].index, 'min_faixa'] = inicio
							continuar = True
		return continuar

	def calAUCRange(self, label, poli, d):	
		inter_points, inter_dict = self.intersections(label, poli)
		
		# calcula o erro estimado para a função de cada grupo selecionado
		erroFaixa = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
		erroFaixa = erroFaixa.astype({'min_faixa': 'float64', 'max_faixa': 'float64' ,'AUC': 'float64'})

		for attr, data in label.groupby(['Atributo']):
			# seleciona o conjunto de limite do atributo
			limites = sorted(set(list(data['minValue'].values) + list(data['maxValue'].values)))
			L = sorted(set(inter_points[attr] + limites))
			
			# calcula os ranges iniciais
			for i in range(len(L)-1):
				inicio = L[i]
				fim = L[i+1]
				xzinho = self.get_groups(inicio, fim, data, d, poli[attr], erroFaixa, attr, 'front', False)
			#print(erroFaixa)
			
			# calcula as extensões dos ranges
			start_points = erroFaixa[(erroFaixa['Atributo'] == attr)]['min_faixa'].values
			end_points = erroFaixa[(erroFaixa['Atributo'] == attr)]['max_faixa'].values
			points = sorted(set(list(start_points) + list(end_points)))
			
			for point in points:
				# pega os grupos cujas funções passam pelo ponto
				clusters =list(data[(data['minValue'] < point)
				          & (data['maxValue'] > point)]['Cluster'].values)
				
				if clusters:
					# pega o início da função mais distante do ponto
					inicio_ = data[data['Cluster'].isin(clusters)]['minValue'].min()
					fim_ = point
					I = sorted(set(np.round(np.linspace(inicio_, fim_, 100),2)))
					# para cada pequeno pedaço, atribui aos clusters
					for i in range(len(I), 1, -1):
						if not self.get_groups(I[i-2], I[i-1], data, d, poli[attr], erroFaixa, attr, 'back', True): 
							break
					
					# pega o fim da função mais distante do ponto
					inicio_ = point 
					fim_ = data[data['Cluster'].isin(clusters)]['maxValue'].max()
					I = sorted(set(np.round(np.linspace(inicio_, fim_, 100),2)))
					# para cada pequeno pedaço, atribui aos clusters
					for i in range(len(I)-1):
						if not self.get_groups(I[i], I[i+1], data, d, poli[attr], erroFaixa, attr, 'front', True): 
							break

			#print(erroFaixa)
		return erroFaixa

	def AUC(self,a, b, func):
		auc, err = integrate.quad(np.poly1d(func),a, b)
		return auc

