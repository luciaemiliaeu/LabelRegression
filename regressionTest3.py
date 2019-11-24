import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

import matplotlib.pyplot as plt
from plotingFunctions import plotRegression, plotPrediction, plotResults
import scipy.integrate as integrate
from math import sqrt
from itertools import combinations
from scipy.optimize import fsolve, newton, broyden1, newton_krylov
import pylab

datasets = [("./databases/sementes.csv",3)]
#("./databases/mnist64.csv",10),("./databases/iris.csv",3),("./databases/vidros.csv",6), ("./databases/sementes.csv",3)]


def erro_metrics(results):
	erroFuncs = pd.DataFrame(columns=['Cluster', 'Atributo', 'RMSE', 'AreaSobCurva'])
	
	for attr, data in results.groupby(['Atributo']):
		for cluster, values in data.groupby(['Cluster']):
			
			attr_column = values.loc[values.index,'Saida'].get_values()
			erro = values.loc[:,'Erro'].get_values()
			poli = np.polyfit(attr_column.astype(float), erro.astype(float), 3)
			
			auc, err = integrate.quad(np.poly1d(poli),min(attr_column), max(attr_column))
			
			erroFuncs.loc[erroFuncs.shape[0],:] = [cluster, attr, rmse, auc ]

def polyApro(results):
	poli = []
	for attr, data in results.groupby(['Atributo']):
		poli.append(([(np.polyfit(values.loc[values.index,'Saida'].get_values().astype(float), values.loc[:,'Erro'].get_values().astype(float), 3), cluster) for cluster, values in data.groupby(['Cluster'])], attr))
	return poli

def AUC(a, b, func):
	auc, err = integrate.quad(np.poly1d(func[0]),a, b)
	return auc

def calErroFaixa(label, faixas, poli):	
	finalLabel = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
	for attr, data in label.groupby(['Atributo']):
		faixas_ = [i[0] for i in faixas if i[1] == attr][0]
		poli_ = [i[0] for i in poli if i[1] == attr][0]
		
		erroFaixa = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
		for i in range(len(faixas_)-1):
			inicio = faixas_[i]
			fim = faixas_[i+1]
			clusters = data[(data['minValue']<= inicio) & (data['maxValue']>=fim)]['Cluster'].get_values()
			for k in clusters:
				erroFaixa.loc[erroFaixa.shape[0],:] = [k, attr, inicio, fim, AUC(inicio, fim, [x[0] for x in poli_ if x[1]==k])]
			
			eminimo = erroFaixa[(erroFaixa['Atributo']==attr) & (erroFaixa['min_faixa']==inicio) & (erroFaixa['max_faixa']==fim)]['AUC'].min()
			clusterFinal = erroFaixa[(erroFaixa['AUC']) == eminimo]
			
			if not clusterFinal.empty:
				if not finalLabel[(finalLabel['Atributo'] == attr)].empty:
					if finalLabel.loc[finalLabel.shape[0]-1,'max_faixa'] == clusterFinal['min_faixa'].get_values()[0] and finalLabel.loc[finalLabel.shape[0]-1,'Cluster'] == clusterFinal['Cluster'].get_values()[0]:
						finalLabel.loc[finalLabel.shape[0]-1,'max_faixa'] = clusterFinal['max_faixa'].get_values()[0]
						finalLabel.loc[finalLabel.shape[0]-1,'AUC'] = (finalLabel.loc[finalLabel.shape[0]-1,'AUC'] + clusterFinal['AUC'].get_values()[0])
					else: finalLabel.loc[finalLabel.shape[0],:] = clusterFinal.get_values()[0]
				else: finalLabel.loc[finalLabel.shape[0],:] = clusterFinal.get_values()[0]
		
	return finalLabel

def calAcerto(faixas, dados):
	#print(faixas.sort_values(by=['Cluster', 'AUC']))
	rotulo = []
	for clt, data in dados.groupby(['classe']):
		regras = faixas[(faixas['Cluster'] == clt)]
		total = data.shape[0]
		acertos = []
		
		for attr in regras['Atributo'].unique():
			for i in regras[(regras['Atributo'] == attr)].index: 				
				regra = regras[(regras['Atributo'] == attr)].loc[i,:]
				n_acertos = data[(data[attr].get_values() >= regra['min_faixa']) & (data[attr].get_values() <= regra['max_faixa'])].shape[0]
				acertos.append((attr, regra['min_faixa'],regra['max_faixa'], n_acertos/total))
		rotulo.append((clt, acertos))
	return rotulo
def findIntersection(fun1,fun2,x0):	
	return fsolve(lambda x : np.polyval(fun1, x) - np.polyval(fun2, x),x0, full_output=True, factor = 10)

def quadratic_intersections(p, q):
    """Given two quadratics p and q, determines the points of intersection"""
    x = np.roots(np.asarray(p) - np.asarray(q))
    y = np.polyval(p, x)
    return x, y
def intersections(faixasAttr, polinomios, minMax):
	x =[]
	for faixas, poly in zip(faixasAttr, polinomios):
		limites = [(i[0],i[1][1]) for i in minMax if i[1][0] == poly[1]]
		combinations_ = list(combinations([i[1] for i in limites], 2))
		for c in combinations_: 	
			x_ = [a[0] for a in limites if (a[1] == c[0] or a[1]==c[1])]
			x_minimo = max([min(a) for a in x_])
			x_maximo = min([max(a) for a in x_])

			if x_minimo < x_maximo:
				step = (x_maximo - x_minimo)/10
				xx = np.linspace(x_minimo,x_maximo, num=10)
				print(x_minimo, x_maximo, step, xx)
				poly1 = [i[0] for i in poly[0] if i[1]==c[0]]
				poly2 = [i[0] for i in poly[0] if i[1]==c[1]]

				for x0 in xx:
					r = findIntersection(poly1[0], poly2[0],x0)
					if (r[3] == 'The solution converged.'):
						if(r[0][0] >= x_minimo and r[0][0]<= x_maximo):
							print(r[0][0])
							x.append((r[0][0], faixas[1], c))
						else: print('out of bounderies')
					else: print('not converged')
				
	x_intersec = []
	for attr, clt in [i[1] for i in minMax]:
		points = [i[0] for i in x if i[1]==attr and clt in i[2]]
		x_intersec.append((np.unique(points).tolist(),attr, clt))
	return x_intersec
	

for dataset, n_clusters in datasets:
	# Extrai o nome da base de dados
	title = dataset.split('/')[2].split('.')[0]+' dataset'
	print("")
	print(title)
	print("")	

	# Cria DataFrame com os valores de X e o cluster Y
	dataset = pd.read_csv(dataset, sep=',',parse_dates=True)
	Y = dataset.loc[:,'classe'].get_values()
	X = dataset.drop('classe', axis=1)
	atributos_names = X.columns
	cluster = pd.DataFrame(X.apply(minmax_scale).get_values(), columns = atributos_names)
	
	error = pd.DataFrame(columns=['Cluster', 'Atributo', 'Saida', 'nor_Saida', 'Erro'])
	label = pd.DataFrame(columns=['Cluster', 'Atributo', 'minValue', 'maxValue', 'Erro'])
	for attr in cluster.columns:
		test_set = cluster.sample(frac=0.33)
		y_test = test_set.loc[:,attr].get_values()
		X_test = test_set.drop(attr, axis=1).get_values()

		train_set = cluster.drop(test_set.index)
		y_train = train_set.loc[:,attr].get_values()
		X_train = train_set.drop(attr, axis=1).get_values()
		

		# Treina o modelo de regressão 
		svr_rbf = SVR(kernel='linear', C=100, gamma='auto')
		svr_rbf.fit(X_train,y_train)
		predicted = svr_rbf.fit(X_train,y_train).predict(X_test)
		
		#plotRegression(cluster.drop(attr, axis=1).get_values(),cluster.loc[:,attr].get_values(), svr_rbf, attr)

		# Dataframe : {y_real, y_Predicted, Cluster, Erro}
		y_ = pd.DataFrame({'Actual': y_test, 'Predicted': predicted, 'Cluster':  Y[test_set.index]})
		y_ = y_.assign(Erro=lambda x: abs(x.Actual-x.Predicted))
		y_.index = test_set.index
		y_ = y_.join(test_set.loc[y_.index, attr])
		
		#plotPrediction(attr, y_)
		
		# Calcula o erro do attr sob a faixa toda
		for c, data in y_.groupby(['Cluster']):
			for out, values in data.groupby([attr]):
				error.loc[error.shape[0],:] = [c, attr, X.loc[values.index[0],attr], out, values.mean(axis=0).Erro]
			rsme = sqrt(mean_squared_error(data.loc[:,'Predicted'], data.loc[:,'Actual']))
			label.loc[label.shape[0],:] = [c, attr, X.loc[data.index, attr].min(), X.loc[data.index, attr].max(), rsme ]	
		#plotResults(title, error)
	polinomios = polyApro(error)
	#print(label.sort_values(by=['Atributo', 'Erro']))

	minMax = [(values[['minValue']].min().get_values().tolist()+values[['maxValue']].max().get_values().tolist(), out) for out, values in label.groupby(['Atributo', 'Cluster'])]
	faixas = [(np.sort(np.unique(values[['minValue', 'maxValue']].get_values())).tolist(), out) for out, values in label.groupby(['Atributo'])]
	intersec = intersections(faixas, polinomios, minMax)
	inter_points = []
	for i in np.unique(label['Atributo'].get_values()).tolist():
		p = [a[0] for a in intersec if a[1]==i]
		a = [x for xs in p for x in xs]
		inter_points.append((a,i))
	
	points = []
	for faixa, inter in zip(faixas, inter_points):
		points.append((np.sort(np.unique(faixa[0] + inter[0])).tolist(), faixa[1]))
	#print(minMax)
	
	plotResults(title, error, polinomios, intersec)
	rotulos = calErroFaixa(label, points, polinomios)
	print(rotulos.sort_values(by=['Cluster', 'AUC']))
	#print(calAcerto(rotulos, dataset))
	#erro_metrics(error)
	#print(label)
	
	
plt.show()