import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

# Read in data files
# function adapted from https://github.com/TharinduDR/
def read_file(path, index='index'):
    originals = []
    translations = []
    z_means = []
    with open(path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            originals.append(row['original'])
            translations.append(row['translation'])
            z_means.append(float(row['z_mean']))

    return pd.DataFrame(
        {'original': originals,
         'translation': translations,
         'z_mean': z_means
         })

# Fit and unfit functions to normalize for sigmoid layer
min_max_scaler = preprocessing.MinMaxScaler()
def fit(df, label):
    x = df[[label]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x)
    df[label] = x_scaled
    return df


def un_fit(df, label):
    x = df[[label]].values.astype(float)
    x_unscaled = min_max_scaler.inverse_transform(x)
    df[label] = x_unscaled
    return df


# Get evaluation metrics
def get_metrics(df_results, output_path=None, dropout=None, lr=None):
    pearson, _ = stats.pearsonr(df_results['pred_zscore'], df_results['z_mean'])
    rmse = mean_squared_error(df_results['z_mean'], df_results['pred_zscore'], squared=False)
    mae = mean_absolute_error(df_results['z_mean'], df_results['pred_zscore'])
    print("Pearson: {}".format(pearson))
    print("RMSE: {}".format(rmse))
    print("MAE: {}".format(mae))

    if output_path != None:
        with open(output_path, 'w') as writer:
            writer.write('lr: {}\n'.format(lr))
            writer.write('dropout: {}\n'.format(dropout))
            writer.write('pearson: {}\n'.format(pearson))
            writer.write('rmse: {}\n'.format(rmse))
            writer.write('mae: {}\n'.format(mae))

