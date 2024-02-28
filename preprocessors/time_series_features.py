import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from pycatch22 import catch22_all
import csv


def extract_features(signal_data, name):
    # Calculate catch22 features
    catch22_results = catch22_all(signal_data)
    catch22_features = catch22_results['values']
    
    features = {
        'name': name,
        'amplitude': np.max(signal_data) - np.min(signal_data),
        'mean': np.mean(signal_data),
        'variance': np.var(signal_data),
        'standard_deviation': np.std(signal_data),
        'rms': np.sqrt(np.mean(signal_data ** 2)),
        'skewness': stats.skew(signal_data),
        'kurtosis': stats.kurtosis(signal_data),
        'median': np.median(signal_data),
        'max': np.max(signal_data),
        'min': np.min(signal_data),
    }
        # Add catch22 features to the dictionary
    for idx, feature_name in enumerate(catch22_results['names']):
        features[feature_name] = catch22_features[idx]

    return features



def read_csv_file(file_name):
    df = pd.read_csv(file_name)
    return df

def df_to_signal(df, output_folder,filename,group_name):
    all_features = []

    for column in df.columns:
        if column != 'frame' and column != 'timestamp' and column!='Image Name' and not column.startswith('Norm'):
            signal_data = df[column].values
            features = extract_features(signal_data, column)
            all_features.append(features)
    with open(f'{output_folder}signal_{group_name}_{filename}', mode='w', newline='') as csv_file:
        fieldnames = list(all_features[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for feature_dict in all_features:
            writer.writerow(feature_dict)

def AUs_to_signal_features(output_folder, input_folder, filename):
    input_file = os.path.join(input_folder, filename)
    df = read_csv_file(input_file)
    df_to_signal(df, output_folder,filename,group_name)


if __name__ == '__main__':
    group_name = 'HC'
    input_folder = f'./csv/asymmetry-featured/{group_name}/'
    output_folder = f'./csv/asymmetry-signal/smile/{group_name}/'
    # List all files in the folder

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the path is a file
        if os.path.isfile(file_path) and not os.path.isfile(f'{output_folder}signal_{group_name}_{filename}'):
            AUs_to_signal_features(output_folder, input_folder, filename)
