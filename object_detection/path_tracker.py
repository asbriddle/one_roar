import numpy as np
import cv2

# path = "data/visem-dataset/videos/1_09.09.02_SSW.avi"

path = "2021-11-21 12-21-00.mkv"
cap = cv2.VideoCapture(path)

corner_track_params = dict(maxCorners = 1000, qualityLevel = 0.25, minDistance = 7, blockSize = 7)
lk_params = dict(maxLevel = 200, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

prevPts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **corner_track_params)

mask = np.zeros_like(prev_frame)

while True:

    # Grab current frame
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)

    good_new = nextPts[status == 1]
    good_prev = prevPts[status == 1]

    # Use ravel to get points to draw lines and circles
    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()

        mask = cv2.line(mask, (int(x_new), int(y_new)), (int(x_prev), int(y_prev)), (0, 255, 0), 3)

        frame = cv2.circle(frame, (int(x_new), int(y_new)), 8, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    prev_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()