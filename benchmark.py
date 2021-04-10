import os
import glob
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset",
	                    type=str,
	                    default=None)
	parser.add_argument("--weight",
	                    type=float,
	                    default=None)
	parser.add_argument("--epoch_min",
	                    type=int,
	                    default=100)
	parser.add_argument("--epoch_max",
	                    type=int,
	                    default=300)
	parser.add_argument("--window_best",
	                    type=int,
	                    default=3)
	args = parser.parse_args()
	return args

def load_dataframe(fname, ckpt):
	colnames=['Epoch', 'Seed', 'ADE', 'FDE', 'COL']
	df = pd.read_csv(fname, header=None, names=colnames, usecols=[0,1,2,3,4])
	df = df.drop_duplicates(subset=['Epoch', 'Seed'], keep='last')
	df = df[df['Epoch'].isin(ckpt)]
	# seed_list = [123]
	# df = df[df['Seed'].isin(seed_list)]
	df.reset_index(drop=True, inplace=True)
	df = df.astype({"Epoch": int, "Seed": int})
	return df

def relative(df_baseline, df_candidate):
	gain_ade = (df_baseline['ADE'] - df_candidate['ADE']) / (df_baseline['ADE'] + 1e-8)
	gain_fde = (df_baseline['FDE'] - df_candidate['FDE']) / (df_baseline['FDE'] + 1e-8)
	gain_col = (df_baseline['COL'] - df_candidate['COL']) / (df_baseline['COL'] + 1e-8)
	return gain_ade, gain_fde, gain_col

def select(df):
	index = (df['ADE'] + df['FDE'] + df['COL']).argmin()
	return index

def compare(foldername, args):

	epoch_list = [epoch for epoch in range(args.epoch_min, args.epoch_max + 10, 10)]

	filevanilla = glob.glob(os.path.join(foldername, '*_0.0000.csv'))[0]
	df_vanilla = load_dataframe(filevanilla, epoch_list)

	if args.weight:

		filename = glob.glob(os.path.join(foldername, '*_{:.4f}.csv'.format(args.weight)))[0]
		df_snce = load_dataframe(filename, epoch_list)

		print('\n----- Detailed Summary -----\n')
		print('Vanilla:\n', df_vanilla)
		print('')
		print('SNCE:\n', df_snce)

		print('\n----- Avg of {:d} Seeds -----\n'.format(len(df_snce['Seed'].unique())))
		print('Vanilla:')
		print(df_vanilla.mean())
		print('')
		print('SNCE:')
		print(df_snce.mean())

	else:
		flist = glob.glob(os.path.join(foldername, '*.csv'))
		flist.sort(key=lambda filename: float(filename.split('_')[-1][:-4]))
		flist.remove(filevanilla)

		print("       \t      \t      \tAverage \t \t Gain \t \t \t \t     Best \t \t \t Gain ")
		print("       \t      \t ---------------------\t --------------------- \t \t -----------------------------\t ---------------------")
		print("Method \tWeight\t  ADE \t  FDE \t  COL \t  ADE \t  FDE \t  COL \t \t Epoch \t  ADE \t  FDE \t  COL \t  ADE \t  FDE \t  COL ")

		idx_best_vanilla = select(df_vanilla)
		best_vanilla = df_vanilla.iloc[idx_best_vanilla, :]
		avg_vanilla = df_vanilla.iloc[max(0, idx_best_vanilla-args.window_best):idx_best_vanilla+args.window_best, :].mean()

		print("Vanilla\t 0.00 \t {:.3f} \t {:.3f} \t {:.3f} \t   x \t   x \t   x \t \t  {:.0f} \t {:.3f} \t {:.3f} \t {:.3f} \t   x \t   x \t   x".format(avg_vanilla['ADE'], avg_vanilla['FDE'], avg_vanilla['COL'] * 100, best_vanilla['Epoch'], best_vanilla['ADE'], best_vanilla['FDE'], best_vanilla['COL'] * 100))

		metric_vanilla = (avg_vanilla['FDE'], avg_vanilla['COL'])
		metric_snce = (float('inf'), float('inf'))	# FDE & COL
		weight_snce = float('nan')

		for filename in flist:

			weight = float(filename.split('_')[-1][:-4])
			df_snce = load_dataframe(filename, epoch_list)

			idx_best_snce = select(df_snce)
			best_snce = df_snce.iloc[idx_best_snce, :]
			avg_snce = df_snce.iloc[max(0, idx_best_snce-args.window_best):idx_best_snce+args.window_best, :].mean()

			gain_best_ade, gain_best_fde, gain_best_col = relative(best_vanilla, best_snce)
			gain_avg_ade, gain_avg_fde, gain_avg_col = relative(avg_vanilla, avg_snce)

			print("S-NCE \t {:.3f}\t {:.3f} \t {:.3f} \t {:.3f} \t {:.1f}%\t {:.1f}%\t {:.1f}%     \t  {:.0f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.1f}%\t {:.1f}%\t {:.1f}%".format(weight, avg_snce['ADE'], avg_snce['FDE'], avg_snce['COL'] * 100, gain_avg_ade * 100, gain_avg_fde * 100, gain_avg_col * 100, best_snce['Epoch'], best_snce['ADE'], best_snce['FDE'], best_snce['COL']  * 100, gain_best_ade * 100, gain_best_fde * 100, gain_best_col * 100))

			if metric_snce[0] + metric_snce[1] * 100 > avg_snce['FDE'] + avg_snce['COL'] * 100:
				metric_snce = (avg_snce['FDE'], avg_snce['COL'])
				weight_snce = weight

		print('Optimal Weight:', weight_snce, '\n')

		return metric_vanilla, metric_snce

def main():
	args = parse_args()

	if args.dataset is None:
		result = list()
		for dataset in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
			print("Dataset:", dataset)
			foldername = 'experiments/pedestrians/models/snce_' + dataset + '_vel'
			(fde_vanilla, col_vanilla), (fde_snce, col_snce) = compare(foldername, args)
			result.append([dataset, fde_vanilla, col_vanilla * 100, fde_snce, col_snce * 100, (1 - col_snce / col_vanilla) * 100 ])
		df = pd.DataFrame(result, columns=['Dataset', 'FDE-Vanilla', 'COL-Vanilla', 'FDE-SNCE', 'COL-SNCE', 'COL-Gain']).set_index('Dataset')
		df.loc['Avg'] = df.mean()
		print(df)
	else:
		print("Dataset:", args.dataset)
		foldername = 'experiments/pedestrians/models/snce_' + args.dataset + '_vel'
		compare(foldername, args)

if __name__ == "__main__":
    main()
