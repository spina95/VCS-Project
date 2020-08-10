import cv2
import matplotlib.pyplot as plt
from painting_detection import painting_detection
from painting_detection2 import painting_detection2

cap = cv2.VideoCapture('Project material/videos/000/VIRB0399.MP4')
frame_rate = 10
frame_count = 0

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while (cap.isOpened()):
    ret, frame = cap.read()
    frame_count += 1

    if ret == True and frame_count % frame_rate == 0:
        pictures = painting_detection(frame)
        # rectified = painting_rectification(pictures)  FRENCI
        # retrieval = painting_retrieval(rectified) HAMID

        frame = cv2.resize(frame, (960, 540))
        cv2.imshow('Contours', frame)

        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)

cap.release()

# Closes all the frames
cv2.destroyAllWindows()