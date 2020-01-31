import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import os

def save_table(dataset_name, table, file_name):
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Testes/'+dataset_name+'/')
	sample_file_name = file_name

	if not os.path.isdir(results_dir):
	    os.makedirs(results_dir)
	tmp = table.select_dtypes(include=[np.number])
	table.loc[:, tmp.columns] = np.round(tmp, 2)
	table.to_csv(results_dir + file_name, index=False)

def save_fig(dataset_name, figName):
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Testes/'+dataset_name+'/')
	sample_file_name = figName

	if not os.path.isdir(results_dir):
	    os.makedirs(results_dir)
	plt.savefig(results_dir + sample_file_name)
