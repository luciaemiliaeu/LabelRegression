import numpy as np
import pandas as pd
import RotulatorModel
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import saving_results as save
import gc
warnings.filterwarnings("ignore")

datasets = ["./databases/breast_cancer.csv","./databases/iris.csv", "./databases/sementes.csv","./databases/wine.csv" ]
#"./databases/breast_cancer.csv","./databases/iris.csv","./databases/vidros.csv", "./databases/sementes.csv","./databases/wine.csv" ]
for dataset in datasets:
	title = dataset.split('/')[2].split('.')[0]
	out = pd.DataFrame(columns =['d', 'accuracys', 'n_elemForLabel'])
	for i in range(10):
		print(title +' '+ str(i))
		d = np.round((i+1)*0.1, 2)
		#parâmetros do rotulados: (dataset, d, t, folds, dataset_name)
		r = RotulatorModel.Rotulator(dataset, (i+1)*0.1, 0.2, 10, title+str(i))
		
		accuracys = r.results['Accuracy'].values
		n_elemForLabel = []
		for clt, data in r.labels.groupby(['Cluster']):
			n = data['Atributo'].unique().shape[0]
			n_elemForLabel.append(n)

		del r
		gc.collect()
		
		out.loc[out.shape[0],['d']]=[d]
		out.loc[out.shape[0]-1,['n_elemForLabel']]=[n_elemForLabel]
		out.loc[out.shape[0]-1,['accuracys']]=[accuracys]
	
	out = out.round(2)
	out.to_csv('./Testes/results_'+title+'.csv', index=False)


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