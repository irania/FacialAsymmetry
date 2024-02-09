import os
import sys
sys.path.insert(0, 'D:/Codes/Python/FacialAsymmetry/helpers')  # Adjust the path as necessary
print(sys.path)
import cv2

import dlib
import numpy as np
from helpers.face import Face
from preprocessors.asymmetry_calculator import process_all_face

def barycentric_coordinates_matrix(all_points, vertex1, vertex2, vertex3):
    # Matrix A for vertices of the triangle, with an extra row for affine transformation
    A = np.array([
        [vertex1[0], vertex2[0], vertex3[0]],
        [vertex1[1], vertex2[1], vertex3[1]],
        [100, 100, 100]
    ])

    # Convert all_points to a 3xN matrix (N = number of points), with an extra row of 1s for affine transformation
    points_matrix = np.vstack((all_points.T, np.ones(all_points.shape[0])))

    # Solve for barycentric coordinates using matrix multiplication
    bary_coords = np.linalg.solve(A, points_matrix)

    return bary_coords[:3, :].T  # Transpose to return an Nx3 matrix (N = number of points)
# Function to compute barycentric coordinates
def barycentric_coordinates(point, vertex1, vertex2, vertex3):
    def triangle_area(v1, v2, v3):
        return abs((v1[0]*(v2[1]-v3[1]) + v2[0]*(v3[1]-v1[1]) + v3[0]*(v1[1]-v2[1])) / 2.0)

    area = triangle_area(vertex1, vertex2, vertex3)
    area1 = triangle_area(point, vertex2, vertex3)
    area2 = triangle_area(point, vertex3, vertex1)
    area3 = triangle_area(point, vertex1, vertex2)

    lambda1 = area1 / area
    lambda2 = area2 / area
    lambda3 = area3 / area

    return lambda1, lambda2, lambda3

def dot_product(v1, v2):
    return np.dot(v1, v2)

def barycentric(p, a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    p = np.array(p)

    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = dot_product(v0, v0)
    d01 = dot_product(v0, v1)
    d11 = dot_product(v1, v1)
    d20 = dot_product(v2, v0)
    d21 = dot_product(v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return u, v, w

# Path to the source directory containing images
src_directory = 'data/images/droopy'  # Update this path
# Path to the destination directory to save processed images
dst_directory = 'data/barycentric/droopy'  # Update this path

# Initialize the facial landmark detector
predictor_path = 'openface_2.2.0/shape_predictor_68_face_landmarks.dat'  # Update this path if necessary
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Define landmark groups for each half of the face
left_side_indices = list(range(0, 8)) + list(range(31, 33)) +list(range(48,51)) +list(range(58,61))+[61,67] + list(range(17, 22)) + list(range(36, 42))  # left jaw,left nose, left brow, left eye
right_side_indices = list(range(16, 8, -1)) + [35,34] + [54,53,52,56,55,64,63,65] + list(range(26, 21, -1)) + [45,44,43,42,47,46]

# Iterate over all images in the source directory
for filename in os.listdir(src_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(src_directory, filename)
        frame = cv2.imread(img_path)
        face_obj = Face(frame)
        aligned_face = face_obj.align_face()
        
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            
            landmarks = predictor(gray, face)

            # Define vertices of the triangle (eyebrows and nose tip)
            left_eyebrow = (landmarks.part(17).x, landmarks.part(17).y)
            right_eyebrow = (landmarks.part(26).x, landmarks.part(26).y)
            nose_tip = (landmarks.part(33).x, landmarks.part(33).y)

             # Extracting and sorting landmarks
            left_side_landmarks = [landmarks.part(n) for n in left_side_indices]
            right_side_landmarks = [landmarks.part(n) for n in right_side_indices]

            # Combine landmarks in desired order
            sorted_landmarks = left_side_landmarks + right_side_landmarks

            # Matrix to store barycentric coordinates for the current frame
            frame_matrix = np.zeros((58, 3))  # 68 landmarks, each with 3 barycentric coordinates

            # Compute barycentric coordinates for each landmark
            for n in range(0, 58):
                point = (sorted_landmarks[n].x, sorted_landmarks[n].y)
                frame_matrix[n] = barycentric(point, left_eyebrow, right_eyebrow, nose_tip)
                
                
            #all_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)])
            #barycentric_matrices = barycentric_coordinates(all_points, left_eyebrow, right_eyebrow, nose_tip)
            # Calculate Asymmetry Index
            m = int(frame_matrix.shape[0] / 2)  # m is half the number of landmarks
            S = np.block([[np.identity(m), np.zeros((m, m))], [np.zeros((m, m)), np.identity(m)]])
            T = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # Adjust this matrix based on your specific symmetry assumption
                
            S_Lambda = np.matmul(S, frame_matrix)
            Lambda_T = np.matmul(frame_matrix, T)
            asymmetry_index = np.linalg.norm(S_Lambda - Lambda_T, 'fro')**2

            #Comput our Asymmetry meathod 
            face = Face(frame)
            aligned_face = face.align_face()
            asymmetry_norm_line, asymmetry_vertically_line, asymmetry_norm_eye, asymmetry_vertically_eye, asymmetry_norm_eyebrow, asymmetry_vertically_eyebrow, asymmetry_norm_mouth, asymmetry_vertically_mouth, asymmetry_norm_nose, asymmetry_vertically_nose, face_width, face_height, mouth_width, mouth_height, nose_width, nose_height, eye_width, eye_height,eyebrow_width, eyebrow_height = process_all_face(face)
            asymmetry_norm_whole = (asymmetry_norm_eye+asymmetry_norm_eyebrow+asymmetry_norm_nose+asymmetry_norm_mouth)/ np.sqrt(face_width**2 + face_height**2)
            asymmetry_vertically_whole= (asymmetry_vertically_eye+asymmetry_vertically_eyebrow+asymmetry_vertically_mouth+asymmetry_vertically_nose) / face_height
            # Display Asymmetry Index on the frame copy
            cv2.putText(frame, f"BAI: {asymmetry_index:.2f}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"EAI: {asymmetry_norm_whole:.2f}", (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"VAI: {asymmetry_vertically_whole:.2f}", (10, 110), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
            print(f"Asymmetry Index: {asymmetry_index:.2f}")


            cv2.imwrite(os.path.join(dst_directory, filename), frame)
    
cv2.destroyAllWindows()