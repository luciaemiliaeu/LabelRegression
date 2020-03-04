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


class Rotulator:
	def __init__(self, dataset, d, t, folds, title):		
		# DataFrames
		# X: atributos, Y: cluster, normalBD: X normalizado
		# attr_names : lista de nomes dos atributos
		self.db, X, Y, normalBD, attr_names = self.importBD(dataset)

		# Constrói os modelos de regressão e retorna um dataframe com as predições
		# predisctions: {'index', 'Atributo', 'predict'}
		models = trainingModels(normalBD, attr_names, title, folds)
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

		# poly : funções polinomiais dos atributos por grupo
		polynomials = self.polyApro(errorByValue)

		# limitPoints: pontos de limite dos intervalos por atributo [([pontos], atributo)]
		# intersecByAttrInCluster: pontos de interseção do atributo em cada grupo [([pontos], atributo, grupo)]	
		# interPointsByAttr: pontos de interseção do atributo [([pontos], atributo)]
		limitPoints, intersecByAttrInCluster, interPointsByAttr = self.rangePatition(polynomials, attrRangeByGroup, attr_names)
		print("limits done")
		
		# calcula a relevância dos intervalos
		# rangeAUC: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'}
		rangeAUC = self.calAUCRange(attrRangeByGroup, limitPoints, polynomials, d)
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
		pltFunc.plot_Func_and_Points(yy, polynomials, intersecByAttrInCluster, title)
		pltFunc.plot_Mean_Points_Erro(errorByValue, title)
		pltFunc.plot_Func_and_PointsMean(errorByValue, polynomials, title)
		pltFunc.plot_Functions(errorByValue, polynomials, title)
		pltFunc.plot_Intersec(errorByValue, polynomials, intersecByAttrInCluster, title)
		pltFunc.plot_AUC(errorByValue, polynomials, rangeAUC, title)
		pltFunc.plot_Limite_Points(errorByValue, polynomials, limitPoints, title)
		
		pltFunc.render_results_table(self.results, title, header_columns=0, col_width=2.0)
		pltFunc.render_labels_table( self.labels, title, header_columns=0, col_width=2.0)
		
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

	def rangePatition(self, polynomials, attrRangeByGroup, attr_names):
		# lista ordenada de valores de inicio  ou final de um polinomio por atributo
		ranges = [(list(np.sort(np.unique(data[['minValue', 'maxValue']].values))), attr) for attr, data in attrRangeByGroup.groupby(['Atributo'])]
		
		# pontos de interseção entre os polinômios ([pontos], attr)
		intersecByAttrInCluster, interPointsByAttr = self.intersections(polynomials, attrRangeByGroup, attr_names)
		
		# todos os pontos "de corte" de um attr: pontos de interseção e pontos de início e fim de faixa
		# ([pontos], attr)
		limitPoints = []
		for i in ranges:
			p = [x[0] for x in interPointsByAttr if x[1]==i[1]]
			a = [np.round(x,2) for xs in p for x in xs]			
			r = [ np.round(elem,2) for elem in i[0] ]
			limitPoints.append((sorted(list(set(r+a))), i[1]))

		return limitPoints, intersecByAttrInCluster, interPointsByAttr
	def intersections(self, polynomials, attrRangeByGroup, attr_names):
		intersections =[]
		# Min e Max de cada atributo por grupo
		minMax = [(list(data[['minValue']].min().values)+list(data[['maxValue']].max().values), attrCluster) for attrCluster, data in attrRangeByGroup.groupby(['Atributo', 'Cluster'])]
		
		# poly = conjunto de polinomios de um attr ([k polinomios], attr)
		for poly in  polynomials:
			# limites do attr poly[1] em cada cluster: ([lim_inf, lim_supe], cluster)
			boundaries = [(i[0],i[1][1]) for i in minMax if i[1][0] == poly[1]]
			# combinação de clusters 2 a 2.
			combinations_ = list(combinations([i[1] for i in boundaries], 2))
			for c in combinations_:
				# calcula o mínimo e o máximo da interseção dos polinomios de c
				x_ = [a[0] for a in boundaries if (a[1] == c[0] or a[1]==c[1])]
				x_minimum = max([min(a) for a in x_])
				x_maximum = min([max(a) for a in x_])
				if x_minimum < x_maximum:
					# divide o intervalo em 10 partes iguais
					step = (x_maximum - x_minimum)/10
					xx = np.linspace(x_minimum,x_maximum, num=10)
					
					#seleciona os polinômios dos grupos c[0] e c[1]
					poly1 = [i[0] for i in poly[0] if i[1]==c[0]]
					poly2 = [i[0] for i in poly[0] if i[1]==c[1]]

					# calcula as interseções em cada intervalo
					for x0 in xx:
						r = fsolve(lambda x : np.polyval(poly1[0], x) - np.polyval(poly2[0], x),x0, full_output=True, factor = 10) 
						if (r[3] == 'The solution converged.' and r[0][0] >= x_minimum and r[0][0]<= x_maximum):
							intersections.append((round(r[0][0],2), poly[1], c))			
		
		intersecByAttrInCluster = []
		for attr, clt in [i[1] for i in minMax]:
			points = [i[0] for i in intersections if i[1]==attr and clt in i[2]]
			intersecByAttrInCluster.append((list(set(points)), attr, clt))
		
		interPointsByAttr = []
		for attr in attr_names:
			p = [a[0] for a in intersecByAttrInCluster if a[1]==attr]
			a = [x for xs in p for x in xs]
			interPointsByAttr.append((a,attr))

		return intersecByAttrInCluster, interPointsByAttr
	
	def calAUCRange(self,label, limites, poli, lim):	
		
		finalrange = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
		finalrange = finalrange.astype({'min_faixa': 'float64', 'max_faixa': 'float64' ,'AUC': 'float64'})

		for attr, data in label.groupby(['Atributo']):
			# seleciona o conjunto de limite do atributo e os polinônios de cada grupo

			limites_ = [i[0] for i in limites if i[1] == attr][0]
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
				clusterFinal = erroFaixa[(erroFaixa['AUC']) <= eminimo + (lim*eminimo)]
				
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

