import os
import numpy as np
import pandas as pd
import re

# Define a function to extract the number from the image name
def extract_number(image_name):
    match = re.search(r'frame_(\d+).jpg', image_name)
    if match:
        return int(match.group(1))
    return None

def sort_df(df):
    # Create a new column with the extracted numbers
    df['ImageNumber'] = df['Image Name'].apply(extract_number)

    # Sort the dataframe based on the 'ImageNumber' column
    df_sorted = df.sort_values(by='ImageNumber')

    # Drop the 'ImageNumber' column if you don't need it anymore
    df_sorted = df_sorted.drop('ImageNumber', axis=1)

    # Reset index if you want it in proper order after sorting
    df_sorted = df_sorted.reset_index(drop=True)
    return df_sorted

def read_csv_file_sort(file_name):
    df = pd.read_csv(file_name)
    df = sort_df(df)
    return df

def read_csv_file(file_name):
    df = pd.read_csv(file_name)
    return df


def asymmetry_feature_extract(output_folder, input_folder, filename):
    input_file = os.path.join(input_folder, filename)
    df = read_csv_file_sort(input_file)

    # Prepare a list to collect columns to be removed
    new_df = pd.DataFrame()
    new_df['Image Name'] = df['Image Name']
    # Initialize the 'whole_norm' and 'whole_vertically' to 0
    new_df['whole_norm'] = 0
    new_df['whole_vertically'] = 0
    # Apply operations for columns related to different parts
    for part in ['line', 'eye', 'eyebrow', 'mouth', 'nose']:
        for metric in ['norm', 'vertically']:
            col_name = f'asymmetry_{metric}_{part}'
            new_col_name = col_name
            
            if part == 'line':
                width_col = 'face_width'
                height_col = 'face_height'
            elif part == 'eyebrow':
                width_col = 'eye_width'
                height_col = 'eye_height'
            else:
                width_col = f"{part}_width"
                height_col = f"{part}_height"
            
            if metric == 'norm':
                new_df[new_col_name] = df[col_name] / np.sqrt(df[width_col]**2 + df[height_col]**2)
                new_df['whole_norm'] += df[col_name]
            
            elif metric == 'vertically':
                new_df[new_col_name] = df[col_name] / df[height_col]
                new_df['whole_vertically'] += df[col_name]

    
    # Normalize 'whole_norm' and 'whole_vertically' by face width and face diameter
    new_df['whole_norm'] = new_df['whole_norm'] / np.sqrt(df['face_width']**2 + df['face_height']**2)
    new_df['whole_vertically'] = new_df['whole_vertically'] / df['face_height']


    # Save the modified DataFrame df to a new CSV file
    output_file = os.path.join(output_folder, filename)
    new_df.to_csv(output_file, index=False)
    return new_df


if __name__ == '__main__':
    #input_folder = './csv/asymmetry/single image/raw/'
    #output_folder = './csv/asymmetry/single image/'
    input_folder = './csv/asymmetry/smile/HC/'
    output_folder = './csv/asymmetry-featured/HC/'
    # List all files in the folder

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the path is a file
        if os.path.isfile(file_path) and not os.path.isfile(output_folder + filename):
            asymmetry_feature_extract(output_folder, input_folder, filename)
