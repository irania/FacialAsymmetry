# This file contains the main function for the video facial asymmetry analysis.
import joblib
import argparse
import glob
import numpy as np
import os
import logging
import pandas as pd
from helpers.crop_align_face import crop_align_face_single_video
from helpers.face import Face
from helpers.image_reader import read_image
from preprocessors.asymmetry_calculator import process_all_face
from preprocessors.time_series_features import df_to_signal
from preprocessors.asymmetry_features import asymmetry_feature_extract


        
logging.basicConfig(level=logging.INFO)

def ensure_directory_exists(directory):
    #Ensure that a given directory exists.
    if not os.path.exists(directory):
        os.mkdir(directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze video facial asymmetry.")
    parser.add_argument("--video_directory", default='./data/web-base', help="Directory containing the video files.")
    # Example of adding another argument
    parser.add_argument("--output_directory", default='./data', help="Output directory for the processed data.")
    args = parser.parse_args()
    
    output_feature_folder = 'D:\Codes\Python\FacialAsymmetry\data\working\csv-unasymmetry-features\PD'
    output_signal_folder = 'D:\Codes\Python\FacialAsymmetry\data\working\csv-unasymmetry-signal\PD\\'
    csvs = glob.glob(os.path.join(output_feature_folder, "*.csv"))
    for csv in csvs:
        df = pd.read_csv(csv)       
        # Calculate the diameter using the formula: sqrt(Height^2 + Width^2)
        df['Diameter'] = np.sqrt(df.iloc[:, -2]**2 + df.iloc[:, -1]**2)

        # Identify columns to normalize (excluding first, second last, last, and 'Diameter')
        columns_to_normalize = df.columns[1:-3]

        # Normalize by Height (second last column)
        normalized_by_height = df.iloc[:, 1:-3].div(df.iloc[:, -3], axis=0)
        normalized_by_diameter = df.iloc[:, 1:-3].div(df.iloc[:, -2], axis=0)

        # Normalize by Diameter
        #normalized_by_diameter = df.iloc[:, 1:-3].div(df['Diameter'], axis=0)

        # Create new DataFrame with the first column unchanged
        new_df = df.iloc[:, [0]].copy()

        # Add normalized columns to the new DataFrame
        for i, col in enumerate(columns_to_normalize):
            new_df[f'Feature{i+1}_norm_by_height'] = normalized_by_height[col]
            new_df[f'Feature{i+1}_norm_by_diameter'] = normalized_by_diameter[col]

        # Drop the 'Diameter' column if not needed in the final DataFrame
        df_to_signal(new_df,output_signal_folder, csv.split('\\')[-1],'test')