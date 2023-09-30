import numpy as np

#define face parts pairs
# Left and right corresponding points, not including the middle vertical line
faces_indices = [(i, 32 - i) for i in range(16)]
eyebrows_indices = [(i, 79 - i) for i in range(33,38)]
eyebrows_bottom_indices = [(i, 88 - i) for i in range(38,42)]
eyest_indices = [(i, 132 - i) for i in range(60,65)]
eyesb_indices = [(i, 140 - i) for i in range(65,68)]
nose_indices = [(55,59),(56,58)]
mouth_indices=[(76,82),(77,81),(78,80),(87,83),(86,84),(88,92),(89,91),(95,93)]
   
#mirror one points to other side
def mirror_point(point, line_point1, line_point2):
    x, y = point
    x1, y1 = line_point1
    x2, y2 = line_point2

    dx, dy = x2 - x1, y2 - y1

    a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
    b = 2 * dx * dy / (dx * dx + dy * dy)

    x_prime = a * (x - x1) + b * (y - y1) + x1
    y_prime = b * (x - x1) - a * (y - y1) + y1

    return np.array([x_prime, y_prime])

#mirror all points
def mirror_points(points, line_point1, line_point2):
    mirrored_points = []
    for point in points:
        mirrored_points.append(mirror_point(point, line_point1, line_point2))
    return np.array(mirrored_points)

#measure asymmetry using mirror_point and normal distance
def measure_asymmetry_euclidean(landmarks,pairs_indices,mid_eyes,mid_mouth,norm_image, right=1):
    mirrored_right_points = mirror_points(landmarks[pairs_indices[:, right]], mid_eyes, mid_mouth)
    diff = np.linalg.norm(landmarks[pairs_indices[:, ~right]] - mirrored_right_points, axis=1)

    asymmetry = np.sum(diff) / norm_image
    return asymmetry

#measure asymmetry using mirror_point and vertically distance
def measure_asymmetry_vertically(landmarks,pairs_indices,mid_eyes,mid_mouth,norm_image, right=1):
    mirrored_right_points = mirror_points(landmarks[pairs_indices[:, right]], mid_eyes, mid_mouth)
    diff = np.abs(landmarks[pairs_indices[:, ~right]][:, 1] - mirrored_right_points[:, 1])

    asymmetry = np.sum(diff) / norm_image
    return asymmetry

def process_image(face, pairs_indices):
    landmarks = face.landmarks

    #landmarks and pairs_indices to numpy array
    landmarks_np = np.array(landmarks)
    pairs_indices_np = np.array(pairs_indices)

    #normalization
    norm_face_width = 1
    norm_face_height = 1

    #measure asymmetry
    asymmetry_norm_right = measure_asymmetry_euclidean(landmarks_np, pairs_indices_np, landmarks[51], landmarks[54], norm_face_width,right=1)
    asymmetry_vertically_right = measure_asymmetry_vertically(landmarks_np, pairs_indices_np, landmarks[51], landmarks[54], norm_face_height,right=1)

    return asymmetry_norm_right, asymmetry_vertically_right

def process_all_face(face):
    asymmetry_norm_line, asymmetry_vertically_line = process_image(face, faces_indices)
    asymmetry_norm_eye, asymmetry_vertically_eye = process_image(face, eyest_indices+eyesb_indices)
    asymmetry_norm_eyebrow, asymmetry_vertically_eyebrow = process_image(face, eyebrows_indices+eyebrows_bottom_indices)
    asymmetry_norm_mouth, asymmetry_vertically_mouth = process_image(face, mouth_indices)
    asymmetry_norm_nose, asymmetry_vertically_nose = process_image(face, mouth_indices)
    
    
    face_width = abs(face.landmarks[32][0] - face.landmarks[0][0])
    face_height = abs(face.landmarks[16][1] - face.landmarks[85][1])
    mouth_width = abs(face.landmarks[82][0] - face.landmarks[76][0])
    mouth_height = abs(face.landmarks[85][1] - face.landmarks[80][1])
    nose_width = abs(face.landmarks[59][0] - face.landmarks[55][0])
    nose_height = abs(face.landmarks[57][1] - face.landmarks[51][1])
    eye_width = abs(face.landmarks[72][0] - face.landmarks[68][0])
    eye_height = abs(face.landmarks[74][1] - face.landmarks[70][1])
    eyebrow_width = abs(face.landmarks[50][0] - face.landmarks[46][0])
    eyebrow_height = abs(face.landmarks[44][1] - face.landmarks[46][1])
    
    return asymmetry_norm_line, asymmetry_vertically_line, asymmetry_norm_eye, asymmetry_vertically_eye, asymmetry_norm_eyebrow, asymmetry_vertically_eyebrow, asymmetry_norm_mouth, asymmetry_vertically_mouth, asymmetry_norm_nose, asymmetry_vertically_nose, face_width, face_height, mouth_width, mouth_height, nose_width, nose_height, eye_width, eye_height,eyebrow_width, eyebrow_height