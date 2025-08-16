import face_recognition
from PIL import Image

# Load images
img1 = face_recognition.load_image_file("B:\\Digital Vote System (Advanced)\\Image\\db_photo.png")
img2 = face_recognition.load_image_file("B:\\Digital Vote System (Advanced)\\Image\\new_photo.png")

# Get face encodings
img1_encodings = face_recognition.face_encodings(img1)
img2_encodings = face_recognition.face_encodings(img2)

# Ensure faces are detected in both images
if len(img1_encodings) == 0:
    print("No face found in the first image.")
elif len(img2_encodings) == 0:
    print("No face found in the second image.")
else:
    img1_encoding = img1_encodings[0]
    img2_encoding = img2_encodings[0]

    # Compare faces with a tolerance level (lower is more strict)
    def compare_faces(encoding1, encoding2, tolerance=0.6):
        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        return distance < tolerance

    # Set a stricter tolerance level to reduce false positives
    match = compare_faces(img1_encoding, img2_encoding, tolerance=0.5)
    if match:
        print("Photo matching")
    else:
        print("Photo not matching")
