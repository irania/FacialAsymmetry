#read images from folder
import cv2


def read_image(file_location):
    image = cv2.imread(file_location)
    return image

#convert image to grayscale
def convert_to_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray