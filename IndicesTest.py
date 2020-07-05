import numpy as np
import pandas as pd
import warnings
import gc
from sklearn.preprocessing import minmax_scale

from regressionModel import trainingModels
from RotulatorModel import rangeDelimitation
from rotulate import Label
import savingResults as save
import plotingFunctions as pltFunc

warnings.filterwarnings("ignore")

def polyApro(results):
	polynomials = {}
	for attr, values in results.groupby(['Atributo']):
		d = {}
		for clt , data in values.groupby(['Cluster']):
			if data.shape[0]>1:
				d[clt] = np.polyfit(data['Actual'].to_numpy().astype(float), data['ErroMedio'].to_numpy().astype(float), 2)
				polynomials[attr] = d

	return polynomials

def importBD(path):
	#Carrega a base de dados e separa os atributos(X) do grupo(Y).
	dataset = pd.read_csv(path, sep=',',parse_dates=True)
	Y = dataset.loc[:,'classe']
	X = dataset.drop('classe', axis=1)
	
	#Normaliza os atributos
	XNormal = pd.DataFrame(X.apply(minmax_scale).values, columns = X.columns)
	XNormal = XNormal.astype('float64')

	#Retorna a base de dados original, os atributos(X), os grupos(Y),
	# os atributos normalizados (XNormal) e a lista de atributos (attr_names)
	return dataset, X, Y, XNormal

def calPredictions(db, X, Y, XNormal):
	# Constrói os modelos de regressão e retorna um dataframe com as predições
	# predisctions: {'index', 'Atributo', 'predict'}
	models = trainingModels( XNormal, 10)
	predictions = models.predictions
	print("regressions done")
	
	#Estrutura de dados para armazenar o erro das predições
	yy = pd.DataFrame(columns= ['Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'])
	for attr in XNormal.columns:
		#seleciona as predições para o atributo attr
		y_ = pd.DataFrame(columns=['Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'])
		y_['Actual'] = X[attr].values
		y_['Normalizado'] = XNormal[attr].values
		y_['Predicted'] = predictions[(predictions['Atributo']==attr)].sort_values(by='index')['predict'].values
		y_['Cluster'] = Y.values
		y_ = y_.assign(Erro=lambda x: abs(x.Normalizado-x.Predicted))	
		y_ = y_.assign(Atributo=attr)

		yy = pd.concat([yy, y_])
	#print(yy.head())
	
	return yy, models

def saveInfoDataset(title, models, yy, errorByValue, polynomials):
	save.save_table(title, models._erros, 'erroRegression.csv')
	save.save_table(title, models._metrics, 'metricsRegression.csv')
	save.save_table(title, yy, 'predictions.csv')
	
	pltFunc.plot_Prediction(title, yy)
	pltFunc.plot_Prediction_Mean_Erro(title, errorByValue)
	pltFunc.plot_Func_and_Points(title, yy, polynomials)
	pltFunc.plot_Mean_Points_Erro(title, errorByValue)
	pltFunc.plot_Func_and_PointsMean(title, errorByValue, polynomials)
	pltFunc.plot_Functions(title, errorByValue, polynomials)

def saveInfoLabel(title, ranged_attr, relevanteRanges, results, labels, rotulation_process, errorByValue, polynomials ):
	save.save_table(title, ranged_attr, 'atributos_ordenados_por_acerto.csv')
	save.save_table(title, relevanteRanges, 'range.csv')
	save.save_table(title, results, 'acuracia.csv')
	save.save_table(title, labels, 'rotulos.csv')
	save.save_table(title, rotulation_process, 'rotulos_por_iteracao.csv')

	pltFunc.plot_AUC(title, errorByValue, polynomials, relevanteRanges)
	pltFunc.render_results_table(title, results, header_columns=0, col_width=2.0)
	pltFunc.render_labels_table(title, labels, header_columns=0, col_width=2.0)
	
	#pltFunc.plot_Intersec(errorByValue, polynomials, intersecByAttrInCluster, title)
	#pltFunc.plot_Limite_Points(errorByValue, polynomials, limitPoints, title)

#datasets = ["./databases/iris.csv"]
datasets = ["./databases/breast_cancer.csv","./databases/vidros.csv", "./databases/sementes.csv","./databases/wine.csv" ]

for dataset in datasets:
	title = dataset.split('/')[2].split('.')[0]
	print(title)

	db, X, Y, XNormal = importBD(dataset)
	yy, models = calPredictions(db, X, Y, XNormal)
	
	errorByValue = (yy.groupby(['Atributo', 'Cluster', 'Actual'])['Erro'].agg({'ErroMedio': np.average})
		.reset_index()
		.astype({'Actual': 'float64', 'ErroMedio': 'float64'}))
		
	attrRangeByGroup = (yy.groupby(['Atributo', 'Cluster'])['Actual'].agg({'minValue': np.min, 'maxValue': np.max})
		.reset_index()
		.astype({'minValue': 'float64', 'maxValue': 'float64'}))
	
	polynomials = polyApro(errorByValue)

	saveInfoDataset(title, models, yy, errorByValue, polynomials)

	out = pd.DataFrame(columns =['d', 'accuracys', 'n_elemForLabel'])
	for i in range(11):
		print(title +' '+ str(i))

		relevanteRanges = rangeDelimitation(attrRangeByGroup, polynomials, i*0.1)
		
		ranged_attr, results, labels, rotulation_process = Label(relevanteRanges, 0.2, db)
		
		saveInfoLabel(title+str(i), ranged_attr, relevanteRanges, results, labels, rotulation_process, errorByValue, polynomials)
		
		out = out.append(pd.Series({'d':np.round(i*0.1,2),
		 'n_elemForLabel': labels.groupby(['Cluster', 'Atributo']).size().values,
		 'accuracys': results['Accuracy'].values}), ignore_index=True)

		del relevanteRanges, ranged_attr, results, labels, rotulation_process
		gc.collect()
		
	out.to_csv('./Teste/results_'+title+'.csv', index=False)
