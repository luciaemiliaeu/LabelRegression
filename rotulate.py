import numpy as np 
import pandas as pd

def calLabel(rangeAUC, V, db):
	# Calcula acurácia dos intervalos
	# accuratedRange: {Cluster, Atributo, min_faixa, min_faixa, Accuracy}
	accuratedRange = rangeAUC.assign(Accuracy=rangeAUC.apply(lambda x: calAccuracyRange(info = x, data=db,classe= x.Cluster), axis=1)).sort_values(by=['Cluster', 'Accuracy'], ascending=[True, False])
	accuratedRange.drop(['AUC'], axis=1, inplace= True)

	labels = pd.DataFrame(columns = accuratedRange.columns)
	labels = labels.astype({'min_faixa': 'float64', 'min_faixa': 'float64', 'Accuracy': 'float64'})
	
	results = pd.DataFrame( columns = ['Cluster', 'Accuracy'])
	results = results.astype({'Accuracy': 'float64'})
	
	rotulation_process = pd.DataFrame(columns = ['Cluster', 'iteracao', 'acuracias'])

	for i in db['classe'].unique():
		# Seleciona todos os pares atributo intervalo candidatos ao rótulo do grupo
		rotulo_ = accuratedRange[(accuratedRange['Cluster']==i)]
		rc = pd.DataFrame(columns=rotulo_.columns)
		idx = rotulo_.index.values.tolist()
		iteracao = 0
		# Adiciona atributos ao rótulo enquanto o acerto em outros grupos for maior que V
		repit = True 
		while repit:
			# seleciona os elementos de maior acurácia
			acc_max = rotulo_.max()['Accuracy']
			pro_attr = rotulo_[(rotulo_['Accuracy'] == acc_max)]
			rotulo_.drop(pro_attr.index, axis=0, inplace=True)
			
			while not pro_attr.empty:
				# calcula o acerto em todos os grupos para os elementos de maior acuracia
				max_other_cluster_acc = []
				for index, row in pro_attr.iterrows():
					acc = acertoRotulo(rc.append(row), db)
					c_ = [x[1] for x in acc if x[0]==i]
					other_c = [x[1] for x in acc if x[0]!=i]
					max_other_cluster_acc.append((index, max(other_c), acc, c_, other_c))
				
				# ordena os elementos pela representação mínima nos outros grupos
				max_other_cluster_acc.sort(key=lambda x: x[1])
				
				# adiciona o próximo par ao rótulo 	
				par = max_other_cluster_acc.pop(0)
				rc = rc.append(pro_attr[(pro_attr.index==par[0])])
				pro_attr.drop(par[0], axis=0, inplace=True)
				rotulation_process.loc[rotulation_process.shape[0],['Cluster', 'iteracao']] = [i,iteracao]
				rotulation_process.loc[rotulation_process.shape[0]-1,['acuracias']] = [par[2]]

				print(rotulation_process)
				input()
				iteracao += 1

				# verifica a restrição
				if par[1]<V or rotulo_.empty:
					repit = False
					break
		
		labels = pd.concat([labels, rc], sort=False)
		results.loc[results.shape[0],:] = [i, [x[1] for x in acc if x[0] == i][0]]

	return results, labels, rotulation_process

def calAccuracyRange(info, data, classe):
	data_ = data[(data['classe'] == classe)][info['Atributo']].values
	acertos = [x for x in data_ if x>=info['min_faixa'] and x<=info['max_faixa']]
	return round(len(acertos) / len(data_),2)

def acertoRotulo(rotulo, data):
	acerto = []
	for clt in data['classe'].unique():
		data_ = data[(data['classe'] == clt)]
		total = data_.shape[0]
		for attr, regras in rotulo.groupby(['Atributo']):
			x = pd.DataFrame(columns = data_.columns)
			for index, row in regras.iterrows():
				x = pd.concat([x,  data_[(data_[attr]>= row['min_faixa']) & (data_[attr]<= row['max_faixa'])]])
			data_ = x
		ac = list(np.round((clt, data_.shape[0]/total),2))
		acerto.append(ac)
	return acerto