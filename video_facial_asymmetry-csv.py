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


         

def load_data_from_single_csv(file_path):
    # Read the CSV file
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

    
    return df_flattened

def extract_distances(video_name, aligned_face_destination, image_files, csv_name):
    data=[]
    for image_file in image_files:
        file_location = os.path.join(aligned_face_destination, image_file)
        try:
        #read image and align it
            img = read_image(file_location)
            face = Face(img)                        
            
            face_data = process_all_face(face)
            data.append([image_file] + list(face_data))
        except Exception as e:
                print(f"Error processing image {video_name} : {e}")

    # Convert data to a DataFrame and then to a CSV file
        df = pd.DataFrame(data, columns=['Image Name', 'asymmetry_norm_line', 'asymmetry_vertically_line', 'asymmetry_norm_eye', 'asymmetry_vertically_eye', 'asymmetry_norm_eyebrow', 'asymmetry_vertically_eyebrow', 'asymmetry_norm_mouth', 'asymmetry_vertically_mouth', 'asymmetry_norm_nose', 'asymmetry_vertically_nose', 'face_width', 'face_height', 'mouth_width', 'mouth_height', 'nose_width', 'nose_height', 'eye_width', 'eye_height', 'eyebrow_width' ,'eyebrow_height'])
    
        df.to_csv(csv_name, index=False)


def asymmetry_analyze(csv_name):

    #step 4: predict
    file_path = f"{csv_name}"  # Naming CSV file according to the sub-directory name
    data = load_data_from_single_csv(file_path)

    # Load model
    loaded_clf = joblib.load('models/asymmetry_model_v3.pkl')
    # Load selected features
    selected_features = joblib.load('models/selected_features_v3.pkl')

    # Filter the prediction data to only include the features the model was trained on
    filtered_data = data[selected_features]

    # Make predictions
    predictions = loaded_clf.predict(filtered_data)
    probabilities = loaded_clf.predict_proba(filtered_data)
    return predictions, probabilities


logging.basicConfig(level=logging.INFO)

def ensure_directory_exists(directory):
    #Ensure that a given directory exists.
    if not os.path.exists(directory):
        os.mkdir(directory)

def process_video(video_path, output_signal_folder):
    #Function to process a single video.
    try:
        predictions, probabilities = asymmetry_analyze(video_path)
        logging.info(f"Video: {video_path} - Predictions: {predictions}, Probabilities: {probabilities}")
        return predictions, probabilities
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze video facial asymmetry.")
    # Example of adding another argument
    parser.add_argument("--output_directory", default='./data-UBC/', help="Output directory for the processed data.")
    args = parser.parse_args()

    

    root_directory = args.output_directory
    output_distances_folder = os.path.join(root_directory, 'csv-distance/')
    output_feature_folder = os.path.join(root_directory, 'csv-features/')
    output_signal_folder = os.path.join(root_directory, 'csv-signal/')
    csvs = glob.glob(os.path.join(output_distances_folder, "*.csv"))
    
    #step 3: extract features
    csv_filenames = [os.path.basename(file) for file in csvs]
    for csv in csv_filenames:
        df = asymmetry_feature_extract(output_feature_folder, output_distances_folder, csv)
    
    csvs = glob.glob(os.path.join(output_signal_folder, "*.csv"))
    # Using the helper function to ensure directories exist
    ensure_directory_exists(root_directory)
    ensure_directory_exists(output_signal_folder)
    
    sum=0
    dw=[]
    for csv in csvs:
        predictions, probabilities =process_video(csv, output_signal_folder)
        sum+=predictions[0]
        if predictions[0]==0:
            dw.append(csv)    
    print(sum)
    print(dw)