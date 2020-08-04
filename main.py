import cv2
import matplotlib.pyplot as plt
from painting_detection import painting_detection
from painting_detection2 import painting_detection2

cap = cv2.VideoCapture('Project material/videos/001/GOPR5818.MP4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        pictures = painting_detection2(frame)
        # rectified = painting_rectification(pictures)  FRENCI
        # retrieval = painting_retrieval(rectified) HAMID
        cv2.imshow('Contours', frame)

        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)

    else:
        break

cap.release()

# Closes all the frames
cv2.destroyAllWindows()