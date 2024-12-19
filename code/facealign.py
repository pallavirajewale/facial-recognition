# USAGE
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
from imutils import paths
import time
import cv2
from skimage.measure import compare_ssim
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

path_to_frame = "C:\\Users\\admin\\Desktop\\Kiosk Python\\Kiosk_demo_3008\\frame.jpg"

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# time.sleep(2.0)


frame = vs.read()
frame = imutils.resize(frame, width=400)

# replace below value with "center" OR "bottom" OR "right" OR "left" for adjusting blue box
box_position = "kiosk"

height, width, channels = frame.shape

if (box_position == "center"):
    (startX1, startY1, endX1, endY1) = (int(width * 0.3),
                                        int(height * 0.25), int(width * 0.7), int(height * 0.75))
if (box_position == "left"):
    (startX1, startY1, endX1, endY1) = (int(width * 0.05),
                                        int(height / 4), int(width * 0.45), int(height * 3 / 4))

if (box_position == "right"):
    (startX1, startY1, endX1, endY1) = (int(width * 0.55),
                                        int(height / 4), int(width * 0.95), int(height * 3 / 4))

if (box_position == "bottom"):
    (startX1, startY1, endX1, endY1) = (int(width * 0.25),
                                        int(height * 0.5), int(width * 0.6), int(height * 0.9))

if (box_position == "kiosk"):
    (startX1, startY1, endX1, endY1) = (int(width * 0.2),
                                        int(height * 0.3), int(width * 0.57), int(height * 0.61))


upper_left1 = (startX1, startY1)
bottom_right1 = (endX1, endY1)

StaticArea = (endX1-startX1)*(endY1-startY1)
position = (10, 50)
withoutMask = 0
mask = 0
startX = 0
startY = 0
endX = 0
endY = 0


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


def mask_detect():
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    global frame
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    global mask, withoutMask, startY, startX, endX, endY
    mask = 0
    withoutMask = 0
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        #label = "Mask" if mask > withoutMask else "No Mask"
        #color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        # cv2.putText(frame, label, (startX, startY - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        #cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    #cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


def get_mask_status():
    while True:
        mask_detect()
        if (mask > withoutMask):
            print("Mask Detected")
            return True
            # break
        if (mask < withoutMask):
            print("No Mask Detected")
            return False
            # break
        if (mask == 0 and withoutMask == 0):
            print("No Face")
            cv2.waitKey(400)


def face_align():
    while True:
        mask_detect()
        global StaticArea, upper_left1, bottom_right1, frame
        upper_left = (startX, startY)
        bottom_right = (endX, endY)
        cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), 2)
        cv2.rectangle(frame, upper_left1, bottom_right1, (255, 0, 0), 2)
        IntersectionArea = max(0, min(endX, endX1) - max(startX, startX1)) * \
            max(0, min(endY, endY1) - max(startY, startY1))
        ImgArea = (endX-startX)*(endY-startY)
        S = ImgArea+StaticArea-IntersectionArea
        ratio = IntersectionArea / S
        print(ratio)
        if(ratio>0.6):
            return True
        cv2.imshow("Frame", frame)
        cv2.waitKey(500)

# print(get_mask_status())

print(get_mask_status())
face_align()
