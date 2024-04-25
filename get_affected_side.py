# This file contains the main function for the video facial asymmetry analysis.
import joblib
import argparse
import glob
import numpy as np
import os
import logging
import pandas as pd
from scipy import stats
from helpers.crop_align_face import crop_align_face_single_video
from helpers.face import Face
from helpers.image_reader import read_image
import numpy as np

#RIGHT_INDEXES = [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,42,43,44,45,46,47,48,49,50,58,59,68,69,70,71,72,73,74,75,80,81,82,83,84,91,92,93,97]
RIGHT_INDEXES = [57,42,43,44,45,46,47,48,49,50,58,59,80,81,82,83,84,91,92,93]
#LEFT_INDEXES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,33,34,35,36,37,38,39,40,41,55,56,60,61,62,63,64,65,66,67,76,77,78,86,87,88,89,95,96]
LEFT_INDEXES =  [57,33,34,35,36,37,38,39,40,41,55,56,76,77,78,86,87,88,89,95]
Nose_Index=0

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def process_face_side(landmarks, previous_landmarks):
    distances = []
    speeds = []
    
    if previous_landmarks is not None:
        # Calculate displacements and speeds for each landmark in the current side
        nose = landmarks[Nose_Index]
        prev_nose = previous_landmarks[Nose_Index]
        for curr_point, prev_point in zip(landmarks, previous_landmarks):
            distance = calculate_distance(curr_point, prev_point)
            distances.append(distance)
            nose_distance = calculate_distance(curr_point, nose)
            nose_distance_prev = calculate_distance(prev_point, prev_nose)
            speeds.append(np.abs(nose_distance-nose_distance_prev))  # Speed is distance per frame, assuming frame rate is constant
    #return mean distances and speeds
    return np.mean(distances), np.mean(speeds)
 


def get_mobility_features(video_name, aligned_face_destination, image_files, csv_name):
    data = []
    previous_landmarks_right = None
    previous_landmarks_left = None
    #sort image files
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for image_file in image_files:
        file_location = os.path.join(aligned_face_destination, image_file)
        try:
            img = read_image(file_location)
            face = Face(img)
            
            # Ensure landmarks are stored in order and accessible by index
            landmarks = {i: landmark for i, landmark in enumerate(face.landmarks)}

            # Process face sides and update previous landmarks
            face_data_right = process_face_side([landmarks[i] for i in RIGHT_INDEXES], previous_landmarks_right)
            previous_landmarks_right = [landmarks[i] for i in RIGHT_INDEXES]

            face_data_left = process_face_side([landmarks[i] for i in LEFT_INDEXES], previous_landmarks_left)
            previous_landmarks_left = [landmarks[i] for i in LEFT_INDEXES]
            
            data.append([image_file] + list(face_data_right) + list(face_data_left))
        except Exception as e:
            print(f"Error processing image {video_name}: {e}")

    # Convert data to a DataFrame and then to a CSV file
    df = pd.DataFrame(data, columns=['Image Name', 'Right Distances', 'Right Speeds', 'Left Distances', 'Left Speeds'])
    df.to_csv(csv_name, index=False)

def extract_features(signal_data, name):
    
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

    return features
#Extract, mean, amplitude, and standard deviation, skewness of the distances and speeds for each side
def get_mobility_statistics(output_sides_folder, csv_name):
    df = pd.read_csv(os.path.join(output_sides_folder, csv_name))
    all_features = []
    for column in df.columns:
        features = extract_features(df[column].values, column)
        all_features.append(features)
        
    return all_features

    
def side_analyze(video_loc, aligned_face_destination, output_sides_folder):
    #extract video name
    video_name= video_loc.split('\\')[-1].split('.')[0]
    video_file= video_loc.split('/')[-1]
    video_loc = '/'.join(video_loc.split('/')[0:-1])
    csv_name = f"{video_name}-output.csv"
    csv_file = f"{output_sides_folder}/{csv_name}"  # Naming CSV file according to the sub-directory name

    if not os.path.exists(csv_file):
        #step 1: crop and align face.
        aligned_face_destination = os.path.join(aligned_face_destination, video_name)

        #step 2: calculate asymmetry
        image_files = [f for f in os.listdir(aligned_face_destination) if f.endswith(('.png', '.jpg', '.jpeg'))] 
        get_mobility_features(video_name, aligned_face_destination, image_files, csv_file)
    all_features = get_mobility_statistics(output_sides_folder, csv_name)
    #if mean distances right > mean distances left, then the left side is affected
    #if mean speeds right > mean speeds left, then the left side is affected
    predictions = []
    probabilities = []
    for feature in all_features:
        if feature['name'] == 'Right Distances':
            mean_right_distances = feature['mean']
        elif feature['name'] == 'Left Distances':
            mean_left_distances = feature['mean']

    if (mean_right_distances > mean_left_distances):
        predictions.append('Left')
        probabilities.append(mean_right_distances - mean_left_distances)   
    else:
        predictions.append('Right')
        probabilities.append(mean_left_distances - mean_right_distances)    
    return predictions, probabilities


logging.basicConfig(level=logging.INFO)

def ensure_directory_exists(directory):
    #Ensure that a given directory exists.
    if not os.path.exists(directory):
        os.mkdir(directory)

def process_video(video_path, aligned_face_destination, output_sides_folder):
    
    #Function to process a single video.
    try:
        predictions, probabilities = side_analyze(video_path, aligned_face_destination, output_sides_folder)
        logging.info(f"Video: {video_path} - Predictions: {predictions}, Probabilities: {probabilities}")
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze video facial asymmetry.")
    parser.add_argument("--video_directory", default='./data/web-base', help="Directory containing the video files.")
    # Example of adding another argument
    parser.add_argument("--output_directory", default='./data/side', help="Output directory for the processed data.")
    args = parser.parse_args()
    video_directory = args.video_directory
    video_directory='\\\\files.ubc.ca\\team\\PPRC\\Camera\\Video Assessment_Atefeh\\booth_txt_happy'
    videos = glob.glob(os.path.join(video_directory, "*.mp4"))

    root_directory = args.output_directory

    aligned_face_destination = '\\\\files.ubc.ca\\team\\PPRC\\Camera\\Video Assessment_Atefeh\\Facial Asymmetry\\csv\\booth_txt_happy\\aligned'
    output_sides_folder = os.path.join(root_directory,'csv-sides')

    # Using the helper function to ensure directories exist
    ensure_directory_exists(root_directory)
    ensure_directory_exists(aligned_face_destination)
    ensure_directory_exists(output_sides_folder)
    # Specify the file path
    file_path = '\\\\files.ubc.ca\\team\\PPRC\\Camera\\Video Assessment_Atefeh\\Video_quality_feed\\side.csv'

    # Read the CSV file
    data_side = pd.read_csv(file_path)
    for video_path in videos:
        print(video_path)
        if(video_path.split('\\')[-1].split('_')[0] not in np.array2string(data_side['ID'].values.astype(int))):
            continue
        process_video(video_path, aligned_face_destination, output_sides_folder)
