import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import scipy.integrate as integrate
from itertools import combinations
from scipy.optimize import fsolve

from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from plotingFunctions import plotPointsMean, plotCurvesPointsMean, plotCurves, plotIntersec, plotPoints, plotResults, plotAUC, plotPoints, plotPointsCurve, plotRegression, plotPrediction, plotPredictionMean, plotLimitePoints,plotData, render_mpl_table
from sklearn.model_selection import GridSearchCV

from regressionModel import trainingModels
from rotulate import calLabel

class Rotulator:
	def __init__(self, dataset, d, t, folds):
		title = dataset.split('/')[2].split('.')[0]
		
		# DataFrames: 
		# X: atributos, Y: cluster, normalBD: X normalizado
		# attr_names : lista de nomes dos atributos
		self.db, X, Y, normalBD, attr_names = self.importBD(dataset)

		# Constrói os modelos de regressão e retorna um dataframe com as predições
		# predisctions: {'index', 'Atributo', 'predict'}
		predictions = trainingModels(normalBD, attr_names, title, folds).predictions
		
		# Estrutura de dados para armazenar o erro das predições
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
		yy = yy.astype({'Actual': 'float64', 'Normalizado': 'float64', 'Predicted': 'float64', 'Erro':'float64'})
		
		# Estrutura de dados pra armazenar o erro médio em cada ponto único do atributo por grupo
		errorByValue = pd.DataFrame(columns=['Cluster', 'Atributo', 'Saida', 'nor_Saida', 'ErroMedio'])
		# Estrutura de dados pra armazenar o início e fim dos atributos em cada grupo
		attrRangeByGroup = pd.DataFrame(columns=['Cluster', 'Atributo', 'minValue', 'maxValue'])

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
		
		# calcula a relevância dos intervalos
		# rangeAUC: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'}
		rangeAUC = self.calAUCRange(attrRangeByGroup, limitPoints, polynomials, d)
		
		# monta os rótulos
		# results: {'Cluster', 'Accuracy'}
		# label: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'}
		self.results, self.label = calLabel(rangeAUC, t, self.db)
		
		plotAUC(errorByValue, polynomials, rangeAUC)

		#plotIntersec(errorByValue, polynomials, inter_points)
		#plt.show()
		'''plotLimitePoints(real_error, poly, rangeAUC, yy)
		
		plotPointsMean(real_error, yy)
		plotCurvesPointsMean(real_error, poly, yy)
		plotCurves(real_error, poly)
		plotIntersec(real_error, poly, inter_points)

		plotResults(title, real_error, poly, inter_points, yy)
		plotAUC(real_error, poly, rangeAUC)
		plotPoints(real_error,yy)
		plotPointsCurve(real_error, poly, inter_points, yy)
		'''
		

		#render_mpl_table('accuracy_'+str(title), result, header_columns=0, col_width=2.0)
		#render_mpl_table('label_'+str(title), label, header_columns=0, col_width=2.0)
	
	def importBD(self, path):
		#trocar nome da classe para Grupo
		dataset = pd.read_csv(path, sep=',',parse_dates=True)
		Y = dataset.loc[:,'classe']
		X = dataset.drop('classe', axis=1)
		attr_names = X.columns
		normalBD = pd.DataFrame(X.apply(minmax_scale).to_numpy(), columns = attr_names)

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
							intersections.append((round(r[0][0],3), poly[1], c))			
		
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
						finalrange.loc[finalrange.shape[0],:] = clusterFinal[(clusterFinal['Cluster']==i)].values
		return finalrange
	def AUC(self,a, b, func):
		auc, err = integrate.quad(np.poly1d(func[0]),a, b)
		return auc


	

	

	