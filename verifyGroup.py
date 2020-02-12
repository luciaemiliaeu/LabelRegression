import pandas as pd 
import numpy as np 
from sklearn import metrics
import warnings
import RotulatorModel
warnings.filterwarnings("ignore")

def obedecem_ao_rotulo(data, rotulos):
	clusters = []
	for clt in rotulos['Cluster'].unique():
		x = data.copy()
		labels = rotulos[(rotulos['Cluster']==clt)]
		for attr, regras in labels.groupby(['Atributo']):
			for index, row in regras.iterrows():
				selec = (x[attr]>= row['min_faixa']) & (x[attr]<=row['max_faixa'])
				x.drop(x[~selec].index, axis=0, inplace=True)
		clusters.append((clt, x.index))
	return clusters

def nao_obedecem_aos_seus_rotulos(data, rotulos):
	clusters = []
	for clt in rotulos['Cluster'].unique():
		not_x = pd.DataFrame(columns = data.columns)
		labels = rotulos[(rotulos['Cluster']==clt)]
		for attr, regras in labels.groupby(['Atributo']):
			for index, row in regras.iterrows():
				selec = (data[attr]>= row['min_faixa']) & (data[attr]<=row['max_faixa']) 
				not_x = pd.concat([not_x, data[(data['classe']==clt) & ~selec]])
		clusters.append((clt, not_x.index))
	return clusters

databases = ['./databases/iris3.csv']
labels = ['./Testes/iris0/rotulos.csv']

for database in databases:
	data = pd.read_csv(database)
	rotulo = RotulatorModel.Rotulator(database, 0.1, 0.7, 10, database.split('/')[2].split('.')[0])
	acc = rotulo.results.mean()['Accuracy']
	#print('Rótulo inicial: ', rotulo.labels)
	print('Acurácia inicial: ', np.round(acc,2))
	IS = metrics.silhouette_score(data.drop(['classe'], axis=1), data['classe'])
	BD = metrics.davies_bouldin_score(data.drop(['classe'], axis=1), data['classe'])
	print('Indices :', IS, BD)
	
	epoc = 0
	iteracao = 0
	while True:
		grupos_segundo_rotulo = obedecem_ao_rotulo(data, rotulo.labels)
		estao_no_grupo_errado = nao_obedecem_aos_seus_rotulos(data, rotulo.labels)

		print('grupo errado: ', estao_no_grupo_errado)
		print(grupos_segundo_rotulo)
		changed = False
		for clt, elemento_errado in estao_no_grupo_errado:
			for e in elemento_errado:
				grupo_certo = [x[0] for x in grupos_segundo_rotulo if x[0]!= clt and e in x[1]]
				if grupo_certo:
					changed = True
					data.loc[e,'classe'] = grupo_certo[0]
					print('grupo certo: ', e, grupo_certo)

		if not changed: 
			print('nenhum elemento mudou de grupo')
			break

		IS = metrics.silhouette_score(data.drop(['classe'], axis=1), data['classe'])
		BD = metrics.davies_bouldin_score(data.drop(['classe'], axis=1), data['classe'])
		
		print('novos indices :', IS, BD)

		data.to_csv('database'+str(iteracao)+'.csv', index = False)
		rotulo = RotulatorModel.Rotulator('database'+str(iteracao)+'.csv', 0.1, 0.1, 10, database.split('/')[2].split('.')[0])
		#print('novo rótulo: ',rotulo.labels)
		print('nova acurácia: ',np.round(rotulo.results.mean()['Accuracy'],2))

		if np.round(rotulo.results.mean()['Accuracy'],2) == 1.0: break
		if np.round(rotulo.results.mean()['Accuracy'],2) == np.round(acc,2): epoc += 1
		if np.round(rotulo.results.mean()['Accuracy'],2) > np.round(acc,2): epoc = 0
		if epoc == 5 : break

		iteracao+=1
		acc = rotulo.results.mean()['Accuracy']
		print('\n\n')