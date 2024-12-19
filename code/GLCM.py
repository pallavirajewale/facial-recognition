import cv2
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

# Initialize Tkinter to open file dialog
tk.Tk().withdraw()  # Hide the root window
video_path = askopenfilename(title="Select a Video", filetypes=[("Video files", "*.mp4;*.avi")])

# Check if a file was selected
if not video_path:
    print("No video selected. Exiting program.")
else:
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for GLCM analysis
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to 8-bit unsigned byte format
        gray_frame_ubyte = img_as_ubyte(gray_frame)
        
        # Calculate GLCM matrix
        glcm = greycomatrix(gray_frame_ubyte, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
        
        # Extract GLCM properties
        contrast = greycoprops(glcm, 'contrast')
        correlation = greycoprops(glcm, 'correlation')
        energy = greycoprops(glcm, 'energy')
        homogeneity = greycoprops(glcm, 'homogeneity')
        
        # Print GLCM features for each frame
        print(f"Frame {frame_count}:")
        print("Contrast:", contrast.mean())
        print("Correlation:", correlation.mean())
        print("Energy:", energy.mean())
        print("Homogeneity:", homogeneity.mean())
        
        # Display the frame with GLCM values as overlay text (optional)
        cv2.putText(frame, f"Contrast: {contrast.mean():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Correlation: {correlation.mean():.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Energy: {energy.mean():.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Homogeneity: {homogeneity.mean():.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Show frame
        cv2.imshow('GLCM Features on Video Frame', frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
