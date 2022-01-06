import numpy as np
import cv2

# path = "data/visem-dataset/videos/1_09.09.02_SSW.avi"
path = "2021-11-21 12-21-00.mkv"
cap = cv2.VideoCapture(path)

object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 5)

while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape
    mask = object_detector.apply(frame)


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(mask, cv2.THRESH_BINARY_INV, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # cv2.putText(frame, 'Sperm', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.4, (36, 255, 12), 2)
            print(f"x: {x}, y: {y}, w: {w}, h: {h}")

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)

    if key == 27:

        break


cap.release()
cv2.destroyAllWindows()


