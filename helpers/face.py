import dlib
import cv2
import numpy as np
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

class Face:
    def __init__(self, image, landmarks_detector_spiga=None):

        # Load the image
        self.face_image = image
        self.gray_face_image = cv2.cvtColor(self.face_image, cv2.COLOR_BGR2GRAY)
        
        # Load the face landmarks if they aren't loaded
        if not hasattr(Face, 'face_predictor'):
            face_landmark_path = '../OpenFace_2.2.0/shape_predictor_68_face_landmarks.dat'
            Face.face_predictor = dlib.get_frontal_face_detector()

        if not hasattr(Face, 'landmark_detector'):
            if(landmarks_detector_spiga is None):
                dataset = 'wflw'
                processor = SPIGAFramework(ModelConfig(dataset))
                Face.landmark_detector = processor.inference
            else:
                Face.landmark_detector = landmarks_detector_spiga

        # Detect face landmarks and crop face
        face_rect = self._find_face_rectangle()
        self.face_image = self._crop_face(face_rect)
        self._set_face_features()
        

    #find face rectangle
    def _find_face_rectangle(self):
        detector = Face.face_predictor
        rects = detector(self.gray_face_image, 1)
        return rects[0] 
    
    #crop face
    def _crop_face(self,rect):
        #return cv2.resize(image,(1400,800))
        # convert rect to (x, y, w, h) format
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        # add some padding
        x, w = x - int(w * 0.1), w + int(w * 0.3)
        y, h = y - int(h * 0.1), h + int(h * 0.3)
        if x<0: 
            x=0
        if y<0:
            y=0
        return self.face_image[y:y + h, x:x + w]
    
    def update_image(self,image):
        self.face_image = image
        self.gray_face_image = cv2.cvtColor(self.face_image, cv2.COLOR_BGR2GRAY)
        self._set_face_features()

    #extract landmarks
    def set_landmarks(image):
        features = Face.landmark_detector(image,[[0,0,image.shape[0],image.shape[1]]])
        return features['landmarks'][0],features['headpose'][0]
    
    def align_face(self):
        # Compute center of gravity (average) for both eyes' landmarks
        left_eye_center =  np. array(self.landmarks[96])
        right_eye_center = np. array(self.landmarks[97])

        # Compute the scale factor between the desired distance and current distance between eye centers
        eyes_center = ((left_eye_center + right_eye_center) / 2)
        eyes_dx = right_eye_center[0] - left_eye_center[0]
        eyes_dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(eyes_dy, eyes_dx)) 
        eyes_dist = np.sqrt(eyes_dx ** 2 + eyes_dy ** 2)
        desired_dist = 0.3 * self.face_image.shape[1]  # Desired distance between eye centers, approximately 30% of the image width
        scale = desired_dist / eyes_dist

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        desiredLeftEye = [0.35,0.35]
        tX = self.face_image.shape[1] * 0.5
        tY = self.face_image.shape[0] * desiredLeftEye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply transformation to image
        output_size = (self.face_image.shape[1], self.face_image.shape[0])
        aligned_face = cv2.warpAffine(self.face_image, M, output_size,flags=cv2.INTER_CUBIC)

        return aligned_face



    def _compute_points(self):
        landmarks = self.landmarks
        # Center of eyes
        left_eye_center = landmarks[96]
        right_eye_center = landmarks[97]
        
        # Eyes midpoints
        eyes_mid_point = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)
        
        # Lip midpoint
        lip_mid_point = ((landmarks[88][0] + landmarks[92][0]) / 2, (landmarks[88][1] + landmarks[92][1]) / 2)
        
                
        return left_eye_center, right_eye_center, eyes_mid_point, lip_mid_point
    
    def _set_face_features(self):
        self.landmarks, self.head_pose = Face.set_landmarks(self.face_image)
        # Compute midpoints and other attributes
        self.left_eye_center, self.right_eye_center, self.eyes_mid_point, self.lip_mid_point = self._compute_points()