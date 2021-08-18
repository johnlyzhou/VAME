import cv2 as cv
from tqdm import tqdm


def crop(path, x_bounds=None, y_bounds=None, sessions=1):
    capture = cv.VideoCapture(path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv.CAP_PROP_FPS))

    if x_bounds:
        width = x_bounds[1] - x_bounds[0]
    else:
        width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        x_bounds = (0, width)
    if y_bounds:
        height = y_bounds[1] - y_bounds[0]
    else:
        height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        y_bounds = (0, height)

    for session in range(sessions):
        out = cv.VideoWriter('{}_{}_{}.mp4'.format(width, height, session + 1), fourcc, fps,
                             (width, height))

        for _ in tqdm(range((frame_count // sessions) * session,
                            (frame_count // sessions) * (session + 1)),
                      disable=not True, desc='Resizing session {} of video'.format(session + 1)):

            ret, frame = capture.read()
            if ret:
                cropped = frame[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1], :]
                out.write(cropped)
            else:
                break
        out.release()

    capture.release()
    cv.destroyAllWindows()


def resize(path, w_factor, h_factor, sessions=1):
    capture = cv.VideoCapture(path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv.CAP_PROP_FPS))

    out_width = int(width * w_factor)
    out_height = int(height * h_factor)

    for session in range(sessions):
        out = cv.VideoWriter('{}_{}_{}.mp4'.format(out_width, out_height, session + 1), fourcc, fps,
                             (out_width, out_height))

        for _ in tqdm(range((frame_count // sessions) * session,
                            (frame_count // sessions) * (session + 1)),
                      disable=not True, desc='Resizing session {} of video'.format(session + 1)):

            ret, frame = capture.read()

            if ret:
                resized = cv.resize(frame, (out_width, out_height))
                out.write(resized)
            else:
                break
        out.release()

    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    width_factor = 0.5
    height_factor = 0.5
    parallel_sessions = 10
    vid_path = '/Users/johnzhou/code/dlc_videos/B125_precon_top.avi'
    video = cv.VideoCapture(vid_path)
    # crop to right half of the video (minus some)
    width_bounds = (int(video.get(cv.CAP_PROP_FRAME_WIDTH))//2+10, int(video.get(cv.CAP_PROP_FRAME_WIDTH)))

    crop(vid_path, x_bounds=width_bounds, sessions=parallel_sessions)
    # resize(PATH, WIDTH_FACTOR, HEIGHT_FACTOR)
