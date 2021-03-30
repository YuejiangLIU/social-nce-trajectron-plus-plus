import os
import glob
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset",
	                    type=str,
	                    default='univ')
	parser.add_argument("--weight",
	                    type=float,
	                    default=None)
	parser.add_argument("--last_num",
	                    type=int,
	                    default=3)
	args = parser.parse_args()
	return args

def load_dataframe(fname, last_num):
	colnames=['Epoch', 'Seed', 'ADE', 'FDE', 'COL']
	ckpt_list = [100 - i * 10 for i in range(last_num)]
	df = pd.read_csv(fname, header=None, names=colnames, usecols=[0,1,2,3,4])
	df = df.drop_duplicates(subset=['Epoch', 'Seed'], keep='last')
	df = df[df['Epoch'].isin(ckpt_list)]
	# seed_list = [123]
	# df = df[df['Seed'].isin(seed_list)]
	df = df.sort_values(by=['Seed'])
	df.reset_index(drop=True, inplace=True)
	df['COL'] *= 100
	return df

def compare(df_baseline, df_candidate):
	gain_ade = (df_baseline['ADE'].mean() - df_candidate['ADE'].mean()) / df_baseline['ADE'].mean()
	gain_fde = (df_baseline['FDE'].mean() - df_candidate['FDE'].mean()) / df_baseline['FDE'].mean()
	gain_col = (df_baseline['COL'].mean() - df_candidate['COL'].mean()) / df_baseline['COL'].mean()
	return gain_ade, gain_fde, gain_col

def main():
	args = parse_args()
	print(args.dataset)

	foldername = 'experiments/pedestrians/models/snce_' + args.dataset + '_vel'

	filevanilla = glob.glob(os.path.join(foldername, '*_0.0000.csv'))[0]
	df_vanilla = load_dataframe(filevanilla, args.last_num)

	if args.weight:
		filename = glob.glob(os.path.join(foldername, '*_{:.4f}.csv'.format(args.weight)))[0]
		df_snce = load_dataframe(filename, args.last_num)

		print('\n----- Detailed Summary -----\n')
		print('Vanilla:\n', df_vanilla)
		print('SNCE:\n', df_snce)

		print('\n----- Avg of {:d} Seeds -----\n'.format(len(df_snce['Seed'].unique())))
		print('Vanilla:')
		print(df_vanilla.mean())
		print('')
		print('SNCE:')
		print(df_snce.mean())
		print('')

		gain_ade, gain_fde, gain_col = compare(df_vanilla, df_snce)
		print("Weight: {:.4f} \t ADE: {:.2f}%, FDE: {:.2f}%, COL: {:.2f}%".format(args.weight, gain_ade * 100, gain_fde * 100, gain_col * 100))

	else:
		flist = glob.glob(os.path.join(foldername, '*.csv'))
		flist.sort(key=lambda filename: float(filename.split('_')[-1][:-4]))
		flist.remove(filevanilla)

		print('\n----- Brief Summary -----\n')
		print("Method \tWeight\t  ADE \t  FDE \t  COL")
		print("Vanilla\t 0.00 \t {:.3f} \t {:.3f} \t {:.3f}".format(df_vanilla['ADE'].mean(), df_vanilla['FDE'].mean(), df_vanilla['COL'].mean()))

		for filename in flist:

			weight = float(filename.split('_')[-1][:-4])
			df_snce = load_dataframe(filename, args.last_num)
			
			print("S-NCE \t {:.2f}\t {:.3f} \t {:.3f} \t {:.3f}".format(weight, df_snce['ADE'].mean(), df_snce['FDE'].mean(), df_snce['COL'].mean()))

			# gain_ade, gain_fde, gain_col = compare(df_vanilla, df_snce)
			# print("Weight: {:8.4f} \t ADE: {:.2f}%\t FDE: {:.2f}%\t COL: {:.2f}%".format(weight, gain_ade.mean() * 100, gain_fde.mean() * 100, gain_col.mean() * 100))


if __name__ == "__main__":
    main()