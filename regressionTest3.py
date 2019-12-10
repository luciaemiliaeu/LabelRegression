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
from plotingFunctions import plotRegression, plotPrediction, plotResults, plotPredictionMean
from sklearn.model_selection import GridSearchCV

from regressionModel import trainingModels
from rotulate import calLabel

datasets = [("./databases/iris.csv",4)]

#("./databases/mnist64.csv",10),("./databases/iris.csv",3),("./databases/vidros.csv",6), ("./databases/sementes.csv",3)]


def polyApro(results):
	poli = []
	for attr, data in results.groupby(['Atributo']):
		poli.append(([(np.polyfit(values.loc[values.index,'Saida'].to_numpy().astype(float), values.loc[:,'Erro'].to_numpy().astype(float), 2), cluster) for cluster, values in data.groupby(['Cluster']) if values.shape[0]>1], attr))
	return poli

def AUC(a, b, func):
	auc, err = integrate.quad(np.poly1d(func[0]),a, b)
	return auc

def calAUCRange(label, faixas, poli, lim):	
	
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
				erroFaixa.loc[erroFaixa.shape[0],:] = [k, attr, inicio, fim, AUC(inicio, fim, [x[0] for x in poli_ if x[1]==k])]
			
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

def findIntersection(fun1,fun2,x0):	
	return fsolve(lambda x : np.polyval(fun1, x) - np.polyval(fun2, x),x0, full_output=True, factor = 10)

def intersections(polynomials, minMax):
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
					r = findIntersection(poly1[0], poly2[0],x0)
					if (r[3] == 'The solution converged.' and r[0][0] >= x_minimum and r[0][0]<= x_maximum):
						x.append((round(r[0][0],3), poly[1], c))			
	x_intersec = []
	for attr, clt in [i[1] for i in minMax]:
		points = [i[0] for i in x if i[1]==attr and clt in i[2]]
		x_intersec.append((points, attr, clt))
	return x_intersec

def result(attr, y_test, y_Predicted, cluster):
	# y_ : {y_real, y_Predicted, Cluster, Erro}
	y_ = pd.DataFrame({'Actual': y_test.to_numpy(), 'Predicted': y_Predicted, 'Cluster': cluster[y_test.index]})
	y_ = y_.assign(Erro=lambda x: abs(x.Actual-x.Predicted))
	y_.index = y_test.index
	y_ = y_.join(attr[y_test.index])

	return y_

def rangePatition(error, label, attr_names ):
	polynomials = polyApro(error)

	# MinMax de cada atributo por grupo
	minMax = [(values[['minValue']].min().to_numpy().tolist()+values[['maxValue']].max().to_numpy().tolist(), out) for out, values in label.groupby(['Atributo', 'Cluster'])]
	
	# lista ordenada de valores no inicio  ou final de um polinomio por atributo
	ranges = [(np.sort(np.unique(values[['minValue', 'maxValue']].to_numpy())).tolist(), out) for out, values in label.groupby(['Atributo'])]
	
	# pontos de interseção dos polinomios ([pontos], attr)
	inter_points = []
	intersec = intersections(polynomials, minMax)
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

def importBD(path):
	dataset = pd.read_csv(path, sep=',',parse_dates=True)
	Y = dataset.loc[:,'classe']
	X = dataset.drop('classe', axis=1)
	attr_names = X.columns
	normalBD = pd.DataFrame(X.apply(minmax_scale).to_numpy(), columns = attr_names)

	return dataset, X, Y, normalBD, attr_names

for dataset, n_clusters in datasets:
	# Extrai o nome da base de dados
	title = dataset.split('/')[2].split('.')[0]+' dataset'
	print("")
	print(title)
	print("")	

	# Cria DataFrame com os valores de X,  o cluster Y e X normalizado
	# retorna o array de nomes da bd
	db, X, Y, normalBD, attr_names = importBD(dataset)

	models = trainingModels(Y, normalBD, attr_names, 2)
	
	real_error = pd.DataFrame(columns=['Cluster', 'Atributo', 'Saida', 'nor_Saida', 'Erro'])
	range_error = pd.DataFrame(columns=['Cluster', 'Atributo', 'minValue', 'maxValue', 'RSME'])

	yy = pd.DataFrame(columns= ['Actual', 'Predicted', 'Atributo','Cluster', 'Erro', 'Saida'])
	for attr in attr_names:
		
		y = normalBD[attr]
		x = normalBD.drop(attr, axis=1)
		
		# Treina o modelo de regressão 
		model = [x[1] for x in models.models[0] if x[0] == attr][0]
		y_Predicted = model.predict(x)
		#plotRegression(x,y, model, attr)

		# y_ : {y_real, y_Predicted, Cluster, Erro}
		y_ = result(normalBD[attr], y, y_Predicted, Y)

		yy = pd.concat([yy, y_[['Actual', 'Predicted', 'Cluster', 'Erro']].assign(Atributo=attr).assign(Saida = lambda x:X.loc[x.index, attr])], sort=True)
		#plotPrediction(attr, y_)	
		
		for clt, data in y_.groupby(['Cluster']):
			for out, values in data.groupby([attr]):
				real_error.loc[real_error.shape[0],:] = [clt, attr, X.loc[values.index[0],attr], out, values.mean(axis=0).Erro]
			rsme = mean_squared_error(data['Predicted'], data['Actual'])
			range_error.loc[range_error.shape[0],:] = [clt, attr, X.loc[data.index, attr].min(), X.loc[data.index, attr].max(), rsme ]		
		#plotPredictionMean(attr, real_error)
	
	poly, points, inter_points = rangePatition(real_error, range_error, attr_names)
	plotResults(title, real_error, poly, inter_points, yy)

	rangeAUC = calAUCRange(range_error, points, poly, 0.2)
	result, label = calLabel(rangeAUC, 0.2, db)
	print(label)
	print(result)
	
plt.show()