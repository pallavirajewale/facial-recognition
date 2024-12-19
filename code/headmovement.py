import cv2
import mediapipe as mp
 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
 
cap = cv2.VideoCapture(0)
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
    results = face_mesh.process(rgb_frame)
 
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
 
    cv2.imshow("Face Mesh", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
 