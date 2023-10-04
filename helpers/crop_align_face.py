import os
import cv2
import dlib
import numpy as np
import numpy as np
import cv2
import pandas as pd
from helpers.face import Face
from helpers.image_reader import read_image



def crop_align_face(input_directory, output_directory):

    # Loop through each video file in the directory
    for video_file in os.listdir(input_directory):

        video_path = os.path.join(input_directory, video_file)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(output_directory, video_name)
        
        if not video_file.endswith(('.mp4', '.avi', '.mkv')) or os.path.exists(output_folder):
            continue  # Skip non-video files

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Process the frame
                face = Face(frame)
                aligned_face = face.align_face()

                # Save the aligned face
                output_image_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')

                # Assume aligned_face is in the format compatible with cv2
                cv2.imwrite(output_image_path, aligned_face)

            except Exception as e:
                print(f"Error processing frame {frame_count} in video {video_file}: {e}")

            frame_count += 1

        cap.release()

    cv2.destroyAllWindows()


def crop_align_face_picture_folder(picture_directory, output_directory):
    image_files = [f for f in os.listdir(picture_directory) if f.endswith(('.png', '.jpg', '.jpeg'))] 
    for image_file in image_files:
        file_location = os.path.join(picture_directory, image_file)
        try:
            #read image and align it
            img = read_image(file_location)
            face = Face(img)

            aligned_face = face.align_face()
            face.update_image(aligned_face)
                            # Save the aligned face
            output_image_path = os.path.join(output_directory, image_file)

                # Assume aligned_face is in the format compatible with cv2
            cv2.imwrite(output_image_path, aligned_face)

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            
    cv2.destroyAllWindows()

def crop_align_face_single_video(video_file, input_directory, output_directory):

    # Loop through each video file in the directory
    video_path = os.path.join(input_directory, video_file)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_directory, video_name)
        
#    if not video_file.endswith(('.mp4', '.avi', '.mkv')) or os.path.exists(output_folder):
#        return  # Skip non-video files

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
                # Process the frame
            face = Face(frame)
            aligned_face = face.align_face()

            # Save the aligned face
            output_image_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')

            # Assume aligned_face is in the format compatible with cv2
            cv2.imwrite(output_image_path, aligned_face)

        except Exception as e:
            print(f"Error processing frame {frame_count} in video {video_file}: {e}")

        frame_count += 1

    cap.release()

    cv2.destroyAllWindows()
