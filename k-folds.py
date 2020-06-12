import pandas as pd 
import os
from sklearn import model_selection

if __name__ == '__main__':
	base_path = './data'
	df = pd.read_csv(os.path.join(base_path, 'train.csv'))
	df['kfold'] = -1
	df = df.sample(frac = 1).reset_index(drop = True)
	y = df.target.values
	kf = model_selection.StratifiedKFold(n_splits = 10)
	for fold_, (_,_) in enumerate(kf.split(X = df, y = y)):
		df.loc[:,'kfold'] = fold_
	df.to_csv(os.path.join(base_path, 'train_folds.csv'), index = False)



