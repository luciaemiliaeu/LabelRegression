import numpy as np
import pandas as pd
import RotulatorModel
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

datasets = ['./databases/parkinson.csv']
#"./databases/mnist64.csv","./databases/iris.csv","./databases/vidros.csv", "./databases/sementes.csv"]
for dataset in datasets:
	#dataset, d, t, folds
	for i in range(10):
		r = RotulatorModel.Rotulator(dataset, (i+1)*0.1, 0.15, 10)
		print(r.label)
		print(r.results)
	plt.show()


'''
print(label)
	print(r.results)
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

'''