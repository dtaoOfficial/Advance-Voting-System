import sys
import cv2
import numpy as np
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        sys.exit(1)
    return image

def detect_face(image):
    modelFile = r"C:\\Users\\mahes\\OneDrive\\Documents\\Projects\\advanced voting system\\Photo Matching\\Photo Matching\\res10.caffemodel"
    configFile = r"C:\\Users\\mahes\\OneDrive\\Documents\\Projects\\advanced voting system\\Photo Matching\\Photo Matching\\deploy.prototxt.txt"
    print(f"Model file: {modelFile}")
    print(f"Config file: {configFile}")

    # Check if the config file exists
    if not os.path.exists(configFile):
        print(f"Config file not found: {configFile}")
        sys.exit(1)

    # Print the first 10 lines of the configFile during runtime
    try:
        with open(configFile, 'r') as f:
            lines = f.readlines()
            print("First 10 lines of the deploy.prototxt file:")
            for i in range(min(10, len(lines))):
                print(lines[i].strip())
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Adjust confidence threshold for stricter detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            face = image[y:y1, x:x1]
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            return gray_face
    return None

def compare_faces(face1, face2):
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("Error: cv2.face module not found. Make sure you have 'opencv-contrib-python' installed.")
        sys.exit(1)

    # Convert labels to numpy array
    labels = np.array([0])
    face_recognizer.train([face1], labels)
    label, confidence = face_recognizer.predict(face2)
    print(f"Prediction confidence: {confidence}")
    return confidence < 50.088  # Set threshold for stricter matching

if len(sys.argv) != 3:
    print("Usage: python verify_photo.py <db_photo_path> <new_photo_path>")
    sys.exit(1)

db_photo_path = sys.argv[1]
new_photo_path = sys.argv[2]

db_image = load_image(db_photo_path)
new_image = load_image(new_photo_path)

db_face = detect_face(db_image)
new_face = detect_face(new_image)

if db_face is None:
    print("No face found in the first image.")
elif new_face is None:
    print("No face found in the second image.")
else:
    if compare_faces(db_face, new_face):
        print("Face verified")
    else:
        print("Face not verified")
