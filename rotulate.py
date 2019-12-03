import numpy as np 
import pandas as pd

def calAccuracyRange(info, data):
	data_ = data[(data['classe'] == info['Cluster'])][info['Atributo']].to_numpy().tolist()
	acertos = [x for x in data_ if x>=info['min_faixa'] and x<=info['max_faixa']]
	return len(acertos) / len(data_)
	
def calLabel(rangeAUC, V, db):
	labels = rangeAUC.assign(Accuracy=rangeAUC.apply(lambda x: calAccuracyRange(info = x, data=db), axis=1))
	maxRankLabels = [(c, i.max()['Accuracy']) for c, i in labels.groupby(['Cluster'])]
	labels_=pd.DataFrame(columns=labels.columns)
	for a in maxRankLabels:
		labels_ = pd.concat([labels_, labels[(labels['Accuracy']>= a[1]-V) & (labels['Cluster']==a[0])]])
	return labels_

def LabelAccuracy(label, data):
	labelsEval = pd.DataFrame(columns=['Cluster', 'Accuracy'])
	frames = pd.DataFrame(columns=data.columns)
	for clt, values in label.groupby(['Cluster']):
		data_ = data[(data['classe'] == clt)]
		total = data_.shape[0]
		for attr, regra in values.groupby('Atributo'):
			data_ = data_[(data_[regra['Atributo']].to_numpy()>= regra['min_faixa'].to_numpy()) & (data_[regra['Atributo']].to_numpy()<= regra['max_faixa'].to_numpy())]#[regra['Atributo']].to_numpy()
		labelsEval.loc[labelsEval.shape[0],:] = [clt, data_.shape[0]/total]
		frames = pd.concat([frames,data_])
	return labelsEval, frames
