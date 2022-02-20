import cv2
import sys

from progress.bar import Bar
import os

# based on https://www.geeksforgeeks.org/face-detection-using-cascade-classifier-using-opencv-python/
def extract_face(src, dest):
    img = cv2.imread(src)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_rect = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10)

    for (x, y, w, h) in face_rect:
        face_img = img[y:y+h, x:x+h]
        face_img = cv2.resize(face_img, (100, 100))
        cv2.imwrite(dest, face_img)
        break
    
def main():
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    files = os.listdir(input_dir)

    progress_bar = Bar("Processing", max=len(files))
    for filename in files:
        src = os.path.join(input_dir, filename)
        dest = os.path.join(output_dir, filename)
        extract_face(src, dest)
        progress_bar.next()

if __name__ == "__main__":
    main()
