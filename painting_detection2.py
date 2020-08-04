import cv2
import matplotlib.pyplot as plt
import numpy as np

def painting_detection2(frame):

    # Shadow removal
    # https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb_planes = cv2.split(frame2)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    # Histogram equalization
    #https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
    equ = cv2.equalizeHist(result_norm)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(equ, h=10, templateWindowSize=7, searchWindowSize=21)

    # Thresold
    ret2,th2 = cv2.threshold(denoised,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Dilation
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(th2, kernel, iterations=1)

    # Closing
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    # Erosion
    erosion = cv2.erode(closing, kernel, iterations=1)
    erosion = 255 - erosion

    # Find frames contours
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    good = []
    areas = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        temp = frame[y:y + h, x:x + w]
        var = np.var(temp)
        if var > 800:   # Remove ROI with low variance
            good.append(c)
            areas.append(w * h)

    mean = np.mean(areas)
    good2 = []
    for i, c in enumerate(good):
        x, y, w, h = cv2.boundingRect(c)
        if w * h > mean * 0.1:  # Remove ROI with small size
            good2.append(c)

    print(good2)
    frames = []
    for i, c in enumerate(good2):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frames.append(frame[y:y+h, x:x+w])


    print(len(contours))
    plt.subplot(121), plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
    plt.subplot(122), plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()
    return frames
