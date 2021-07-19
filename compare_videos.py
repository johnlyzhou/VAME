import os
import cv2 as cv
import numpy as np

FRAME_RATE=30.0
CROP_SIZE=(600, 300)

no_tail = cv.VideoCapture('/Users/johnzhou/code/VAME/videos/aligned_video.avi')
tail = cv.VideoCapture('/Users/johnzhou/code/VAME/videos/aligned_video_full_tail.avi')
frame_count = int(no_tail.get(cv.CAP_PROP_FRAME_COUNT)/10)

fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('videos/test_aligned_video.avi', fourcc, FRAME_RATE, CROP_SIZE)

for _ in range(frame_count):
    _, frame_a = no_tail.read()
    _, frame_b = tail.read()
    frame = np.append(frame_a, frame_b, axis=1)
    out.write(frame)

no_tail.release()
tail.release()
out.release()
