import numpy as np
import pandas as pd
import RotulatorModel
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import saving_results as save
import gc
warnings.filterwarnings("ignore")

datasets = ["./databases/iris.csv"]
#"./databases/breast_cancer.csv","./databases/iris.csv","./databases/vidros.csv", "./databases/sementes.csv","./databases/wine.csv" ]
for dataset in datasets:
	title = dataset.split('/')[2].split('.')[0]
	out = pd.DataFrame(columns =['d', 'accuracys', 'n_elemForLabel'])
	
	for i in range(11):
		print(title +' '+ str(i))

		#parâmetros do rotulados: (dataset, d, t, folds, dataset_name)
		r = RotulatorModel.Rotulator(dataset, i*0.1, 0.2, 10, title)
		
		accuracys = r.results['Accuracy'].values
		n_elemForLabel = []
		for clt, data in r.labels.groupby(['Cluster']):
			n = data['Atributo'].unique().shape[0]
			n_elemForLabel.append(n)

		del r
		gc.collect()
		
		out.loc[out.shape[0],['d']]=[i*0.1]
		out.loc[out.shape[0]-1,['n_elemForLabel']]=[n_elemForLabel]
		out.loc[out.shape[0]-1,['accuracys']]=[accuracys]
	
	out = out.round(2)
	out.to_csv('./Testes/results_'+title+'.csv', index=False)
