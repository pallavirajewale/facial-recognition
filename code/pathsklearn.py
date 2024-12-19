import cv2
print(cv2.__version__)  # Check the version
recognizer = cv2.face.LBPHFaceRecognizer_create()
print("LBPH Recognizer created successfully")


