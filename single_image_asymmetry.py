import os
from preprocessors.asymmetry_features import asymmetry_feature_extract
from video_facial_asymmetry import extract_distances


aligned_face_destination = 'Z:\Video Assessment_Atefeh\Facial Asymmetry\csv\\booth_txt_happy\\single-happy\HC'
csv_distance_folder = 'D:\Codes\Python\FacialAsymmetry\data\working\csv-single\\'
output_feature_folder = 'D:\Codes\Python\FacialAsymmetry\data\working\csv-single\\'
file_name = 'Happy-Booth-HC-distance.csv'
#step 2: calculate asymmetry
image_files = [f for f in os.listdir(aligned_face_destination) if f.endswith(('.png', '.jpg', '.jpeg'))]  # Assuming these extensions, adjust as needed
extract_distances('', aligned_face_destination, image_files, os.path.join(csv_distance_folder, file_name))
            
#step 3: extract features
df = asymmetry_feature_extract(output_feature_folder, csv_distance_folder, file_name)