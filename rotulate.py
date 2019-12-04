import numpy as np 
import pandas as pd

def calAccuracyRange(info, data, classe):
	data_ = data[(data['classe'] == classe)][info['Atributo']].values
	acertos = [x for x in data_ if x>=info['min_faixa'] and x<=info['max_faixa']]
	return len(acertos) / len(data_)
	
def calLabel(rangeAUC, V, db):
	labels = rangeAUC.assign(Accuracy=rangeAUC.apply(lambda x: calAccuracyRange(info = x, data=db,classe= x.Cluster), axis=1)).sort_values(by=['Cluster', 'Accuracy'], ascending=[True, False])
	rotulo = pd.DataFrame(columns = labels.columns)
	result = pd.DataFrame( columns = ['Cluster', 'Accuracy'])
	for i in db['classe'].unique():
		attrs_cluster = labels[(labels['Cluster']==i)]
		idx = attrs_cluster.index.values.tolist()
		rc = rotulo[(rotulo['Cluster']==i)]
		
		# Adiciona atributos ao rótulo enquanto o acerto em outros grupos for maior que V
		repit = True 
		while repit:			
			rc = pd.concat([rc, attrs_cluster[(attrs_cluster.index==idx.pop(0))]], sort=False)
			acc = acertoRotulo(rc, db)
			c_ = [x[1] for x in acc if x[0]==i]
			other_c = [x[1] for x in acc if x[0]!=i]
			if all([x<=V for x in other_c]) or not idx: 
				repit = False
		result.loc[result.shape[0],:] = [i, [x[1] for x in acc if x[0] == i][0]]
		rotulo = pd.concat([rotulo, rc], sort=False)
	return result, rotulo

def acertoRotulo(rotulo, data):
	acerto = []
	for clt in data['classe'].unique():
		data_ = data[(data['classe'] == clt)]
		total = data_.shape[0]
		for index, row in rotulo.iterrows():
			data_ = data_[(data_[row['Atributo']]>= row['min_faixa']) & (data_[row['Atributo']]<= row['max_faixa'])]
		acerto.append((clt, data_.shape[0]/total))
	return(acerto)