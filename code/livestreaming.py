import cv2

# Replace these with your Tapo camera credentials and IP
camera_ip = "192.168.1.68"  # Replace with your camera's IP address
username = "easyparkai"  # RTSP username from Tapo app
password = "easypark@123"  # RTSP password from Tapo app

# RTSP URL format for Tapo cameras
#rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/stream1"

# rtsp://admin:admin@192.168.1.200:554/avstream/channel=1/stream=0.sdp

# Open the video stream
cap = cv2.VideoCapture(f"rtsp://admin:admin@192.168.1.200:554/avstream/channel=1/stream=0.sdp")

if not cap.isOpened():
    print("Error: Unable to open video stream. Check your RTSP URL or camera settings.")
else:
    print("Streaming video from Tapo camera. Press 'q' to exit.")

# Display the video feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    cv2.imshow("Tapo Camera Live Stream", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
