import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import seaborn as sns
import pandas as pd

from collections import OrderedDict

def pretty_dataset_name(dataset_name):
    if dataset_name == 'eth':
        return 'ETH - Univ'
    elif dataset_name == 'hotel':
        return 'ETH - Hotel'
    elif dataset_name == 'univ':
        return 'UCY - Univ'
    elif dataset_name == 'zara1':
        return 'UCY - Zara 1'
    elif dataset_name == 'zara2':
        return 'UCY - Zara 2'
    else:
        return dataset_name

def summarize(foldername, keyword, metrics):
	print(f'\n---- {metrics} ----\n')

	dataset_names = ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'Average']
	alg_name = 'Ours'

	perf_df = pd.DataFrame()
	for dataset in dataset_names:
	    for f in glob.glob(f"{foldername}/{dataset}_{keyword}*{metrics}.csv"):
	        dataset_df = pd.read_csv(f)
	        dataset_df['dataset'] = dataset
	        dataset_df['method'] = alg_name
	        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
	        del perf_df['Unnamed: 0']

	    if dataset != 'Average':
	        print(f"{pretty_dataset_name(dataset)}: \t {perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean():.4f}")
	    else:
	        print(f"{pretty_dataset_name(dataset)}: \t {perf_df[(perf_df['method'] == alg_name)]['value'].mean():.4f}")

def main():
	
	foldername = 'results/vel'
	keyword = 'vel_12'

	metrics = 'ade_best_of'
	summarize(foldername, keyword, metrics)
	metrics = 'fde_best_of'
	summarize(foldername, keyword, metrics)
	metrics = 'col_best_of'
	summarize(foldername, keyword, metrics)

	metrics = 'ade_most_likely'
	summarize(foldername, keyword, metrics)
	metrics = 'fde_most_likely'
	summarize(foldername, keyword, metrics)
	metrics = 'col_most_likely'
	summarize(foldername, keyword, metrics)

if __name__ == '__main__':
    main()        