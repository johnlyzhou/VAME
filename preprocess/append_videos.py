import cv2 as cv
from tqdm import tqdm
import numpy as np

PARALLEL_SESSIONS = 10

path = '/Users/johnzhou/code/VAME/results/masked_batches/mask_310_480_1.mp4'
capture = cv.VideoCapture(path)
fourcc = cv.VideoWriter_fourcc(*'mp4v')

width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fps = int(capture.get(cv.CAP_PROP_FPS))
frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

out = cv.VideoWriter('masked_{}_{}.mp4'.format(width, height), fourcc, fps, (width, height))

video = np.zeros((height, width, 3, frame_count * PARALLEL_SESSIONS))

for session in range(1, PARALLEL_SESSIONS + 1):
    path = '/Users/johnzhou/code/VAME/results/masked_batches/mask_{}_{}_{}.mp4'.format(width, height, session)
    print(path)
    capture = cv.VideoCapture(path)

    for fr in tqdm(range(frame_count), disable=not True, desc='Appending session {}'.format(session)):

        ret, frame = capture.read()

        if ret:
            video[:, :, :, fr + frame_count * (session - 1)] = frame.astype('int')

for fr in tqdm(range(frame_count * PARALLEL_SESSIONS), disable=not True, desc='Writing video'):
    out.write(video[:, :, :, fr].astype(np.uint8))

out.release()
