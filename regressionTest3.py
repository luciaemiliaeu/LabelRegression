import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from math import sqrt
from itertools import combinations
from scipy.optimize import fsolve
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from plotingFunctions import plotRegression, plotPrediction, plotResults

datasets = [("./databases/iris.csv",3)]
#("./databases/mnist64.csv",10),("./databases/iris.csv",3),("./databases/vidros.csv",6), ("./databases/sementes.csv",3)]


def polyApro(results):
	poli = []
	for attr, data in results.groupby(['Atributo']):
		poli.append(([(np.polyfit(values.loc[values.index,'Saida'].get_values().astype(float), values.loc[:,'Erro'].get_values().astype(float), 3), cluster) for cluster, values in data.groupby(['Cluster'])], attr))
	return poli

def AUC(a, b, func):
	auc, err = integrate.quad(np.poly1d(func[0]),a, b)
	return auc

def calAUCRange(label, faixas, poli):	
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
	
def partitionDB(X, Y, test_size):
	test_set = X.sample(frac=test_size)
	y_test = test_set.loc[:,Y].get_values()
	X_test = test_set.drop(Y, axis=1).get_values()

	train_set = X.drop(test_set.index)
	y_train = train_set.loc[:,Y].get_values()
	X_train = train_set.drop(Y, axis=1).get_values()

	return test_set, train_set, X_test, X_train, y_test, y_train

def importBD(path):
	dataset = pd.read_csv(path, sep=',',parse_dates=True)
	Y = dataset.loc[:,'classe'].get_values()
	X = dataset.drop('classe', axis=1)
	attr_names = X.columns
	normalBD = pd.DataFrame(X.apply(minmax_scale).get_values(), columns = attr_names)

	return dataset, X, Y, attr_names, normalBD

def trainModel(X_train, y_train, X_test ):
	model = SVR(kernel='linear', C=100, gamma='auto')
	model.fit(X_train,y_train)
	y_predicted = model.fit(X_train,y_train).predict(X_test)
	return model, y_predicted


def result(attr, y_test, y_Predicted, cluster):
	idx = y_test.index
	y = y_test.get_values()

	# y_ : {y_real, y_Predicted, Cluster, Erro}
	y_ = pd.DataFrame({'Actual': y, 'Predicted': y_Predicted, 'Cluster':  cluster[idx]})
	y_ = y_.assign(Erro=lambda x: abs(x.Actual-x.Predicted))
	y_.index = idx
	y_ = y_.join(attr[idx])

	return y_

def rangePatition(error, label, attr_names ):
	polynomials = polyApro(error)

	# MinMax de cada atributo por grupo
	minMax = [(values[['minValue']].min().get_values().tolist()+values[['maxValue']].max().get_values().tolist(), out) for out, values in label.groupby(['Atributo', 'Cluster'])]
	
	# lista ordenada de valores no inicio  ou final de um polinomio por atributo
	ranges = [(np.sort(np.unique(values[['minValue', 'maxValue']].get_values())).tolist(), out) for out, values in label.groupby(['Atributo'])]
	
	# pontos de interseção dos polinomios ([pontos], attr)
	inter_points = []
	intersec = intersections(polynomials, minMax)
	for attr in attr_names:
		p = [a[0] for a in intersec if a[1]==attr]
		a = [x for xs in p for x in xs]
		inter_points.append((a,attr))
	
	# todos os pontos "de corte" de um attr: pontos de interseção e pontos de início e fim de faixa
	# ([pontos], attr)
	points = []
	for range_, inter in zip(ranges, inter_points):
		points.append((np.sort(np.unique(range_[0] + inter[0])).tolist(), range_[1]))
	
	return polynomials, points, intersec

def calAccuracyRange(info, data):
	data_ = data[(data['classe'] == info['Cluster'])][info['Atributo']].get_values().tolist()
	acertos = [x for x in data_ if x>=info['min_faixa'] and x<=info['max_faixa']]
	return len(acertos) / len(data_)
	
def calLabel(rangeAUC, V):
	labels = rangeAUC.assign(Accuracy=rangeAUC.apply(lambda x: calAccuracyRange(info = x, data=db), axis=1))
	maxRankLabels = [(c, i.max()['Accuracy']) for c, i in labels.groupby(['Cluster'])]
	labels_ = labels[(labels['Accuracy']+V >= [a[1] for a in maxRankLabels if a[0]==labels['Cluster'][0]])].sort_values(by=['Cluster', 'Accuracy'], ascending= [True, False])
	return labels_

def LabelAccuracy(label, data):
	labelsEval = pd.DataFrame(columns=['Cluster', 'Accuracy'])
	frames = pd.DataFrame(columns=data.columns)
	for clt, values in label.groupby(['Cluster']):
		data_ = data[(data['classe'] == clt)]
		total = data_.shape[0]
		for attr, regra in values.groupby('Atributo'):
			data_ = data_[(data_[regra['Atributo']].get_values()>= regra['min_faixa'].get_values()) & (data_[regra['Atributo']].get_values()<= regra['max_faixa'].get_values())]#[regra['Atributo']].get_values()
		labelsEval.loc[labelsEval.shape[0],:] = [clt, data_.shape[0]/total]
		frames = pd.concat([frames,data_])
	return labelsEval, frames

for dataset, n_clusters in datasets:
	# Extrai o nome da base de dados
	title = dataset.split('/')[2].split('.')[0]+' dataset'
	print("")
	print(title)
	print("")	

	# Cria DataFrame com os valores de X e o cluster Y
	db, X, Y, attr_names, normalBD = importBD(dataset)
	
	real_error = pd.DataFrame(columns=['Cluster', 'Atributo', 'Saida', 'nor_Saida', 'Erro'])
	range_error = pd.DataFrame(columns=['Cluster', 'Atributo', 'minValue', 'maxValue', 'RSME'])

	for attr in attr_names:
		#y = attr
		test_set, train_set, X_test, X_train, y_test, y_train = partitionDB(normalBD, attr, 0.33)

		# Treina o modelo de regressão 
		model, y_Predicted = trainModel(X_train, y_train, X_test)
		#plotRegression(normalBD.drop(attr, axis=1).get_values(),normalBD.loc[:,attr].get_values(), model, attr)

		# y_ : {y_real, y_Predicted, Cluster, Erro}
		y_ = result(normalBD[attr], normalBD.loc[test_set.index, attr], y_Predicted, Y)
		#plotPrediction(attr, y_)	
		
		for clt, data in y_.groupby(['Cluster']):
			for out, values in data.groupby([attr]):
				real_error.loc[real_error.shape[0],:] = [clt, attr, X.loc[values.index[0],attr], out, values.mean(axis=0).Erro]
			rsme = sqrt(mean_squared_error(data.loc[:,'Predicted'], data.loc[:,'Actual']))
			range_error.loc[range_error.shape[0],:] = [clt, attr, X.loc[data.index, attr].min(), X.loc[data.index, attr].max(), rsme ]		
	
	poly, points, inter_points = rangePatition(real_error, range_error, attr_names)
	
	rangeAUC = calAUCRange(range_error, points, poly)
	label = calLabel(rangeAUC, 0.1)
	result, frames = LabelAccuracy(label, db)

	print(label)
	print(result)
	plotResults(title, real_error, poly, inter_points)
plt.show()