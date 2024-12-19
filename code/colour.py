import cv2 

import numpy as np 

 

def calculate_color_features(frame): 

    # Resize frame for faster processing (optional) 

    resized_frame = cv2.resize(frame, (320, 240)) 

 

    # Convert the frame to the HSV color space for color features 

    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV) 

 

    # Calculate mean color in RGB and HSV 

    mean_bgr = cv2.mean(resized_frame)[:3] 

    mean_hsv = cv2.mean(hsv_frame)[:3] 

 

    # Calculate color histogram (using 16 bins for each channel in HSV) 

    h_hist = cv2.calcHist([hsv_frame], [0], None, [16], [0, 180]) 

    s_hist = cv2.calcHist([hsv_frame], [1], None, [16], [0, 256]) 

    v_hist = cv2.calcHist([hsv_frame], [2], None, [16], [0, 256]) 

 

    # Normalize the histograms 

    h_hist = cv2.normalize(h_hist, h_hist).flatten() 

    s_hist = cv2.normalize(s_hist, s_hist).flatten() 

    v_hist = cv2.normalize(v_hist, v_hist).flatten() 

 

    # Combine histograms into one feature vector 

    color_hist_features = np.concatenate([h_hist, s_hist, v_hist]) 

 

    return mean_bgr, mean_hsv, color_hist_features 

 

# Open a video capture object (0 for default camera) 

cap = cv2.VideoCapture(0) 

 

while True: 

    # Read frame-by-frame 

    ret, frame = cap.read() 

    if not ret: 

        break 

 

    # Calculate color features for the frame 

    mean_bgr, mean_hsv, color_hist_features = calculate_color_features(frame) 

 

    # Display mean color information on the frame 

    color_info = f"Mean BGR: {mean_bgr}, Mean HSV: {mean_hsv}" 

    cv2.putText(frame, color_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 

 

    # Show the frame with the overlayed text 

    cv2.imshow('Video Feed with Color Features', frame) 

 

    # Exit on pressing 'q' 

    if cv2.waitKey(1) & 0xFF == ord('q'): 

        break 

 

# Release resources 

cap.release() 

cv2.destroyAllWindows() 