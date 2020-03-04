import numpy as np
import pandas as pd
import scipy.integrate as integrate
from itertools import combinations
from scipy.optimize import fsolve
from sklearn.preprocessing import minmax_scale

import plotingFunctions as pltFunc
import savingResults as save
from regressionModel import trainingModels
from rotulate import calLabel

import random as rand


class Rotulator:
	def __init__(self, dataset, d, t, folds, title):		
		# DataFrames
		# X: atributos, Y: cluster, normalBD: X normalizado
		# attr_names : lista de nomes dos atributos
		self.db, X, Y, normalBD, attr_names = self.importBD(dataset)

		# Constrói os modelos de regressão e retorna um dataframe com as predições
		# predisctions: {'index', 'Atributo', 'predict'}
		models = trainingModels(d, normalBD, attr_names, title, folds)
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
			y_['Normalizado'] = normalBD[attr].values
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
		
		'''
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
		#pltFunc.plot_Func_and_Points(yy, polynomials, intersecByAttrInCluster, title)
		pltFunc.plot_Mean_Points_Erro(errorByValue, title)
		pltFunc.plot_Func_and_PointsMean(errorByValue, polynomials, title)
		pltFunc.plot_Functions(errorByValue, polynomials, title)
		#pltFunc.plot_Intersec(errorByValue, polynomials, intersecByAttrInCluster, title)
		pltFunc.plot_AUC(errorByValue, polynomials, rangeAUC, title)
		#pltFunc.plot_Limite_Points(errorByValue, polynomials, limitPoints, title)
		
		pltFunc.render_results_table(self.results, title, header_columns=0, col_width=2.0)
		pltFunc.render_labels_table( self.labels, title, header_columns=0, col_width=2.0)
		'''
	def importBD(self, path):
		#trocar nome da classe para Grupo
		dataset = pd.read_csv(path, sep=',',parse_dates=True)
		Y = dataset.loc[:,'classe']
		X = dataset.drop('classe', axis=1)
		attr_names = X.columns
		normalBD = pd.DataFrame(X.apply(minmax_scale).values, columns = attr_names)
		normalBD = normalBD.astype('float64')

		return dataset, X, Y, normalBD, attr_names
	
	def polyApro(self, results):
		poli = []
		for attr, data in results.groupby(['Atributo']):
			poli.append(([(np.polyfit(values.loc[values.index,'Saida'].to_numpy().astype(float), values.loc[:,'ErroMedio'].to_numpy().astype(float), 2), cluster) for cluster, values in data.groupby(['Cluster']) if values.shape[0]>1], attr))
		return poli
	
	def calAUCRange(self, label, poli, d):	
		#label: 'Cluster', 'Atributo', 'minValue', 'maxValue'
		#poli: [([([coef], grupo)], attr)]
		
		finalrange = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
		finalrange = finalrange.astype({'min_faixa': 'float64', 'max_faixa': 'float64' ,'AUC': 'float64'})

		for attr, data in label.groupby(['Atributo']):
			# seleciona o conjunto de limite do atributo e os polinônios de cada grupo
			minimo_ = data.min()['minValue']
			maximo_ = data.max()['maxValue']

			limites_ = np.linspace(minimo_,maximo_, num=500)

			outros_min = data[(data['minValue'] > minimo_)]['minValue'].values
			outros_max = data[(data['maxValue'] < maximo_)]['maxValue'].values

			limites_ = np.insert(limites_,  limites_.shape[0], outros_min)
			limites_ = np.insert(limites_,  limites_.shape[0], outros_max)

			limites_ = np.round(limites_, 2)
			limites_ = np.sort(np.unique(limites_))

			poli_ = [i[0] for i in poli if i[1] == attr][0]

			for i in range(len(limites_)-1):
				# delimita a faixa
				inicio = limites_[i]
				fim = limites_[i+1]	
				# seleciona os grupos com domínio da função do erro na faixa
				clusters = data[(data['minValue']<= inicio) & (data['maxValue']>=fim)]['Cluster'].values

				# calcula o erro estimado para a função de cada grupo selecionado
				erroFaixa = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
				erroFaixa = erroFaixa.astype({'min_faixa': 'float64', 'max_faixa': 'float64' ,'AUC': 'float64'})

				for k in clusters:
					erroFaixa.loc[erroFaixa.shape[0],:] = [k, attr, inicio, fim, self.AUC(inicio, fim, [x[0] for x in poli_ if x[1]==k])]
				
				# calcula o erro mínimo
				eminimo = erroFaixa[(erroFaixa['Atributo']==attr) & (erroFaixa['min_faixa']==inicio) & (erroFaixa['max_faixa']==fim)]['AUC'].min()
				# seleciona os grupos para os quais a faixa será atribuída com base no parâmetro d
				clusterFinal = erroFaixa[(erroFaixa['AUC']) <= eminimo + (d*eminimo)]

				
				# Verifica se é necessário concatenar faixas
				for i in clusterFinal['Cluster'].values:
					min_ = clusterFinal[(clusterFinal['Cluster']==i)]['min_faixa'].values
					max_ = clusterFinal[(clusterFinal['Cluster']==i)]['max_faixa'].values
					auc = clusterFinal[(clusterFinal['Cluster']==i)]['AUC'].values

					if not finalrange[(finalrange['Cluster']==i) & (finalrange['Atributo']==attr) & (finalrange['max_faixa']==min_[0])].empty:
						finalrange.loc[finalrange[(finalrange['Cluster']==i) & (finalrange['Atributo']==attr) & (finalrange['max_faixa']==min_[0])].index, 'AUC'] += auc[0]	
						finalrange.loc[finalrange[(finalrange['Cluster']==i) & (finalrange['Atributo']==attr) & (finalrange['max_faixa']==min_[0])].index, 'max_faixa'] = max_[0]

					else:
						finalrange.loc[finalrange.shape[0],:] = clusterFinal[(clusterFinal['Cluster']==i)].values[0]

		return finalrange
	def AUC(self,a, b, func):
		auc, err = integrate.quad(np.poly1d(func[0]),a, b)
		return auc

