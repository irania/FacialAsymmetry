import os
import sys
import numpy as np
import pandas as pd


def load_data(folder_path, label):
    data_frames = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=None)

            # Extract feature names from first row and first column
            row_labels = df.iloc[0, 1:].values
            col_labels = df.iloc[1:, 0].values


            df = df.apply(pd.to_numeric, errors='coerce')

            # Combine row_labels and col_labels into feature names
            feature_names = [f"{row}_{col}"  for col in col_labels for row in row_labels]

            # Remove the first row and first column after extracting labels
            df = df.iloc[1:, 1:]
            flattened_features = df.values.flatten()

            # Create a DataFrame for the current file's data
            df_flattened = pd.DataFrame([flattened_features], columns=feature_names)

            # Add the label column
            df_flattened['label'] = label
            df_flattened['user'] = file.split('.')[0]

            # Add the current file's DataFrame to the list of DataFrames
            data_frames.append(df_flattened)

    # Concatenate all DataFrames in the list into a single DataFrame
    all_data = pd.concat(data_frames, ignore_index=True)

    return all_data



def create_dataset(folder_path, label1, label2):

    label1_data = load_data(folder_path+label1, 1)
    label2_data = load_data(folder_path+label2, 0)


    return label1_data,label2_data
