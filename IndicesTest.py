import numpy as np
import pandas as pd
import RotulatorModel
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

datasets = ['./databases/sementes.csv']
#("./databases/mnist64.csv",10),("./databases/iris.csv",3),("./databases/vidros.csv",6), ("./databases/sementes.csv",3)]
for dataset in datasets:
	
	r = RotulatorModel.Rotulator(dataset)
	label = r.label
	data = r.db
	print(label)
	print(r.result)
	IS = metrics.silhouette_score(data.drop(['classe'], axis=1), data['classe'])
	BD = metrics.davies_bouldin_score(data.drop(['classe'], axis=1), data['classe'])
	print(IS, BD)
	for clt in data['classe'].unique():
		label_ = label[(label['Cluster']== clt)]
		for attr, regras in label_.groupby(['Atributo']):
			for index, row in regras.iterrows():
				data.drop(data[(~(data[attr]>= row['min_faixa']) & (data[attr]<= row['max_faixa'])) & (data['classe'] == clt)].index, axis=0, inplace=True)
	
	IS = metrics.silhouette_score(data.drop(['classe'], axis=1), data['classe'])
	BD = metrics.davies_bouldin_score(data.drop(['classe'], axis=1), data['classe'])
	print(IS, BD)
