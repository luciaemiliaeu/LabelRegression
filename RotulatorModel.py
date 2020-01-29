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
		# points: pontos de limite dos intervalos 
		# inter_points: pontos de interseção entre as funções	
		poly, points, inter_points = self.rangePatition(errorByValue, attrRangeByGroup, attr_names)

		# calcula a relevância dos intervalos
		# rangeAUC: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'}
		rangeAUC = self.calAUCRange(attrRangeByGroup, points, poly, d)
		
		# monta os rótulos
		# results: {'Cluster', 'Accuracy'}
		# label: {'Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'}
		self.results, self.label = calLabel(rangeAUC, t, self.db)

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
		dataset = pd.read_csv(path, sep=',',parse_dates=True)
		Y = dataset.loc[:,'classe']
		X = dataset.drop('classe', axis=1)
		attr_names = X.columns
		normalBD = pd.DataFrame(X.apply(minmax_scale).to_numpy(), columns = attr_names)

		return dataset, X, Y, normalBD, attr_names

	def rangePatition(self, error, label, attr_names):
		polynomials = self.polyApro(error)

		# MinMax de cada atributo por grupo
		minMax = [(list(data[['minValue']].min().values)+list(data[['maxValue']].max().values), out) for out, data in label.groupby(['Atributo', 'Cluster'])]
		
		# lista ordenada de valores no inicio  ou final de um polinomio por atributo
		ranges = [(list(np.sort(np.unique(data[['minValue', 'maxValue']].values))), out) for out, data in label.groupby(['Atributo'])]
		
		# pontos de interseção dos polinomios ([pontos], attr)
		inter_points = []
		intersec = self.intersections(polynomials, minMax)
		for attr in attr_names:
			p = [a[0] for a in intersec if a[1]==attr]
			a = [x for xs in p for x in xs]
			inter_points.append((a,attr))
		
		# todos os pontos "de corte" de um attr: pontos de interseção e pontos de início e fim de faixa
		# ([pontos], attr)
		all_points = []
		for i in ranges:
			p = [x[0] for x in inter_points if x[1]==i[1]]
			a = [x for xs in p for x in xs]
			all_points.append((np.unique(np.sort(i[0]+a)), i[1]))
		
		return polynomials, all_points, intersec
	def polyApro(self, results):
		poli = []
		for attr, data in results.groupby(['Atributo']):
			poli.append(([(np.polyfit(values.loc[values.index,'Saida'].to_numpy().astype(float), values.loc[:,'ErroMedio'].to_numpy().astype(float), 2), cluster) for cluster, values in data.groupby(['Cluster']) if values.shape[0]>1], attr))
		return poli
	def intersections(self,polynomials, minMax):
		x =[]
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
					
					poly1 = [i[0] for i in poly[0] if i[1]==c[0]]
					poly2 = [i[0] for i in poly[0] if i[1]==c[1]]

					for x0 in xx:
						r = self.findIntersection(poly1[0], poly2[0],x0)
						if (r[3] == 'The solution converged.' and r[0][0] >= x_minimum and r[0][0]<= x_maximum):
							x.append((round(r[0][0],3), poly[1], c))			
		x_intersec = []
		for attr, clt in [i[1] for i in minMax]:
			points = [i[0] for i in x if i[1]==attr and clt in i[2]]
			x_intersec.append((points, attr, clt))
		return x_intersec
	def findIntersection(self,fun1,fun2,x0):	
		return fsolve(lambda x : np.polyval(fun1, x) - np.polyval(fun2, x),x0, full_output=True, factor = 10)
	
	def calAUCRange(self,label, faixas, poli, lim):	
		
		finalLabel = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
		for attr, data in label.groupby(['Atributo']):
			faixas_ = [i[0] for i in faixas if i[1] == attr][0]
			poli_ = [i[0] for i in poli if i[1] == attr][0]

			for i in range(len(faixas_)-1):
				erroFaixa = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
				inicio = faixas_[i]
				fim = faixas_[i+1]
				clusters = data[(data['minValue']<= inicio) & (data['maxValue']>=fim)]['Cluster'].to_numpy()
				for k in clusters:
					erroFaixa.loc[erroFaixa.shape[0],:] = [k, attr, inicio, fim, self.AUC(inicio, fim, [x[0] for x in poli_ if x[1]==k])]
				
				eminimo = erroFaixa[(erroFaixa['Atributo']==attr) & (erroFaixa['min_faixa']==inicio) & (erroFaixa['max_faixa']==fim)]['AUC'].min()
				clusterFinal = erroFaixa[(erroFaixa['AUC']) <= eminimo + (lim*eminimo)]
				for i in clusterFinal['Cluster'].values:
					min_ = clusterFinal[(clusterFinal['Cluster']==i)]['min_faixa'].values
					max_ = clusterFinal[(clusterFinal['Cluster']==i)]['max_faixa'].values
					auc = clusterFinal[(clusterFinal['Cluster']==i)]['AUC'].values
					if not finalLabel[(finalLabel['Cluster']==i) & (finalLabel['Atributo']==attr) & (finalLabel['max_faixa']==min_[0])].empty:
						finalLabel.loc[finalLabel[(finalLabel['Cluster']==i) & (finalLabel['Atributo']==attr) & (finalLabel['max_faixa']==min_[0])].index, 'AUC'] += auc[0]	
						finalLabel.loc[finalLabel[(finalLabel['Cluster']==i) & (finalLabel['Atributo']==attr) & (finalLabel['max_faixa']==min_[0])].index, 'max_faixa'] = max_[0]
					else:
						finalLabel.loc[finalLabel.shape[0],:] = clusterFinal[(clusterFinal['Cluster']==i)].values
		return finalLabel
	def AUC(self,a, b, func):
		auc, err = integrate.quad(np.poly1d(func[0]),a, b)
		return auc


	

	

	
