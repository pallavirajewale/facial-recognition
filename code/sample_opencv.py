import cv2
import face_recognition

# Initialize the webcam (0 is the default camera)
video_capture = cv2.VideoCapture(1)

# Load a sample image and learn how to recognize it.
# You can replace this with any image of the person you want to recognize
known_image = face_recognition.load_image_file("known_person.jpg")  # Change path to the known image
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Create arrays to hold the known face encodings and their names
known_face_encodings = [known_face_encoding]
known_face_names = ["Known Person"]  # Name of the person in known image

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame of video
    # The result is a list of face locations and a list of encodings
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches the known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default name if no match is found

        # If a match was found, use the corresponding name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Label the face with the name
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
