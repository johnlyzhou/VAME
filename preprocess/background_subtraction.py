import cv2 as cv
import numpy as np
from tqdm import tqdm


def get_background(path, num_batches=10, save=True):
    capture = cv.VideoCapture(path)
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    video = np.zeros((height, width, 3, frame_count // num_batches)).astype(int)
    backgrounds = np.zeros((height, width, 3, num_batches)).astype(int)

    for batch_idx in range(num_batches):
        for fr in tqdm(range(frame_count // num_batches),
                       disable=not True, desc='Calculating background for session {} of video'.format(batch_idx + 1)):
            ret, frame = capture.read()
            video[:, :, :, fr] = frame.astype('int')

        for x_pix in range(height):
            for y_pix in range(width):
                for channel in range(3):
                    backgrounds[x_pix, y_pix, channel, batch_idx] = np.argmax(
                        np.bincount(video[x_pix, y_pix, channel, :]))

    # background = np.mean(backgrounds, axis=3)
    # assert background.shape == backgrounds[:, :, :, 0].shape

    if save:
        np.save('results/backgrounds.npy', backgrounds)

    capture.release()
    return backgrounds


def subtract_background(path, background, tolerance=75):
    print('Beginning background subtraction on {}'.format(path))

    capture = cv.VideoCapture(path)
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv.CAP_PROP_FPS))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('mask_{}.mp4'.format(path), fourcc, fps, frame_size)

    # subtract background from each frame
    for _ in tqdm(range(frame_count), disable=not True, desc='Removing background'):
        ret, frame = capture.read()

        if ret:
            for x_pix in range(height):
                for y_pix in range(width):
                    flag = True
                    for ch in range(3):
                        bg = background[x_pix, y_pix, ch]

                        if bg - tolerance > frame[x_pix, y_pix, ch] or frame[x_pix, y_pix, ch] > bg + tolerance:
                            flag = False
                    if flag:
                        frame[x_pix, y_pix, :] = [255, 255, 255]

            frame = frame.astype(np.uint8)
            out.write(frame)

    out.release()
    cv.destroyAllWindows()
    print('Completed background subtraction on {}'.format(path))


def main(path):
    # get_background(path)
    background = np.load('results/background.npy')
    paths = ['/Users/johnzhou/code/VAME/results/batches/310_480_{}.mp4'.format(i) for i in range(6, 11)]
    for path in paths:
        subtract_background(path, background)


if __name__ == '__main__':
    vid_path = '/Users/johnzhou/code/VAME/results/310_480.mp4'
    main(vid_path)
