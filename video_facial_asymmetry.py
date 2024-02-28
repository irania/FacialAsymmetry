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


def asymmetry_analyze(video_loc,aligned_face_destination,output_distances_folder,output_feature_folder,output_signal_folder):
    #extract video name
    video_name= video_loc.split('\\')[-1].split('.')[0]
    video_file= video_loc.split('/')[-1]
    video_loc = '/'.join(video_loc.split('/')[0:-1])
    csv_name = f"{video_name}-output.csv"
    csv_file = f"{output_distances_folder}/{csv_name}"  # Naming CSV file according to the sub-directory name
    if not os.path.exists(csv_file):
        #step 1: crop and align face.
        crop_align_face_single_video(video_file, video_loc, aligned_face_destination)
        aligned_face_destination = os.path.join(aligned_face_destination, video_name)

        #step 2: calculate asymmetry
        image_files = [f for f in os.listdir(aligned_face_destination) if f.endswith(('.png', '.jpg', '.jpeg'))]  # Assuming these extensions, adjust as needed
        extract_distances(video_name, aligned_face_destination, image_files, csv_file)
            
    #step 3: extract features
    file_path = f'{output_signal_folder}/signal_test_{csv_name}'
    if(os.path.exists(file_path)):
        return
    df = asymmetry_feature_extract(output_feature_folder, output_distances_folder, csv_name)
    df_to_signal(df,output_signal_folder, csv_name,'test')
        
    #step 4: predict
    
    data = load_data_from_single_csv(file_path)

    # Load model
    loaded_clf = joblib.load('models/best_model_v3.pkl')
    # Load selected features
    selected_features = joblib.load('models/selected_features_v3.pkl')

    # Filter the prediction data to only include the features the model was trained on
    filtered_data = data[selected_features]

    # Make predictions
    predictions = loaded_clf.predict(filtered_data)
    #probabilities = loaded_clf.predict_proba(filtered_data)
    return predictions, 100#probabilities


logging.basicConfig(level=logging.INFO)

def ensure_directory_exists(directory):
    #Ensure that a given directory exists.
    if not os.path.exists(directory):
        os.mkdir(directory)

def process_video(video_path, aligned_face_destination, output_distances_folder, output_feature_folder, output_signal_folder):
    #Function to process a single video.
    try:
        predictions, probabilities = asymmetry_analyze(video_path, aligned_face_destination, output_distances_folder, output_feature_folder, output_signal_folder)
        logging.info(f"Video: {video_path} - Predictions: {predictions}, Probabilities: {probabilities}")
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze video facial asymmetry.")
    parser.add_argument("--video_directory", default='./data/web-base', help="Directory containing the video files.")
    # Example of adding another argument
    parser.add_argument("--output_directory", default='./data', help="Output directory for the processed data.")
    args = parser.parse_args()
    video_directory = args.video_directory
    #video_directory = 'Z:\Video Assessment_Atefeh\\booth_txt_disgust'
    #video_directory = 'Z:\Video Assessment_Atefeh\Facial Asymmetry\PD'
    videos = glob.glob(os.path.join(video_directory, "*.webm"))

    root_directory = args.output_directory
    #root_directory = f'Z:\Video Assessment_Atefeh\Facial Asymmetry\csv'
    aligned_face_destination = os.path.join(root_directory,'aligned')
    output_distances_folder = os.path.join(root_directory,'csv-distance')
    output_feature_folder = os.path.join(root_directory,'csv-features')
    output_signal_folder = os.path.join(root_directory, 'csv-signal/')

    # Using the helper function to ensure directories exist
    ensure_directory_exists(root_directory)
    ensure_directory_exists(aligned_face_destination)
    ensure_directory_exists(output_distances_folder)
    ensure_directory_exists(output_feature_folder)
    ensure_directory_exists(output_signal_folder)
    
    for video_path in videos:
        process_video(video_path, aligned_face_destination, output_distances_folder, output_feature_folder, output_signal_folder)

    # output_feature_folder = 'D:\Codes\Python\FacialAsymmetry\data\working\csv-features\PD'
    # output_signal_folder = 'D:\Codes\Python\FacialAsymmetry\data\working\csv-signal\PD\\'
    # csvs = glob.glob(os.path.join(output_feature_folder, "*.csv"))
    # for csv in csvs:
    #     df = pd.read_csv(csv)       
    #     df_to_signal(df,output_signal_folder, csv.split('\\')[-1],'test')