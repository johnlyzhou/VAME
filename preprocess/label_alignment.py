import numpy as np
import pandas as pd
import tqdm


# If a video is cropped using the crop() method in alter_video, this method will shift the DLC tracking points to match
# e.g. if you cropped off the left half of a (640, 480) video to (320, 480), you would set left_bound=320 and each point
# would have 320 subtracted off its x-coordinate.
def write_cropped_dlc_labels(df, left_bound=0, bottom_bound=0, save_file=None):
    if save_file is None:
        save_file = 'results/cropped_dlc.csv'

    frames, body_parts = (23560, 5)

    for f in tqdm.tqdm(range(frames), disable=not True, desc='Writing points'):
        for b in range(body_parts):
            df.iloc[f + 2, b * 3 + 1] = float(df.iloc[f + 2, b * 3 + 1]) - left_bound
            df.iloc[f + 2, b * 3 + 2] = float(df.iloc[f + 2, b * 3 + 2]) - bottom_bound

    df.to_csv(save_file, index=False)


# If a video is resized (downsampled) using the resize() method in alter_video, this method will contract the DLC
# tracking points to match, e.g. if you downsample a (640, 480) video to (320, 240), then this method will divide
# x- and y-coordinates of each point by 2.
def write_resized_dlc_labels(df, old_dims, new_dims, pts, save_file=None):
    if save_file is None:
        save_file = 'results/resized_dlc.csv'

    width_factor = new_dims[0]/old_dims[0]
    height_factor = new_dims[1]/old_dims[1]

    frames, body_parts, coords = pts.shape

    for f in tqdm.tqdm(range(frames), disable=not True, desc='Writing points'):
        for b in range(body_parts):
            df.iloc[f + 2, b * 3 + 1] = float(df.iloc[f + 2, b * 3 + 1]) * width_factor
            df.iloc[f + 2, b * 3 + 2] = float(df.iloc[f + 2, b * 3 + 2]) * height_factor

    df.to_csv(save_file, index=False)


# If a video is cropped and aligned using the VAME egocentrical_alignment() method and the shifted DLC points are saved,
# this method can be used to replace the coordinates in a DLC-formatted CSV with the new aligned coordinates.
def write_shifted_dlc_labels(pts, df, save_file=None):
    if save_file is None:
        save_file = 'results/shifted_dlc.csv'

    frames, body_parts, coords = pts.shape

    for f in tqdm.tqdm(range(frames), disable=not True, desc='Writing points'):
        for b in range(body_parts):
            df.iloc[f + 2, b * 3 + 1] = pts[f, b, 0]
            df.iloc[f + 2, b * 3 + 2] = pts[f, b, 1]

    df.to_csv(save_file, index=False)


if __name__ == '__main__':
    # original = pd.read_csv('/Users/johnzhou/code/dlc_videos'
    #                        '/B125_precon_topDLC_resnet101_FearConditioningTopNov27shuffle1_950000filtered.csv',
    #                        low_memory=False)
    # write_cropped_dlc_labels(original, left_bound=330)

    # now run egocentrical_alignment() with these new cropped points

    points = np.load('/Users/johnzhou/code/VAME/results/dlc_points.npy')
    csv = pd.read_csv('/Users/johnzhou/code/VAME/results/310_480_masked.csv',
                      low_memory=False)
    write_shifted_dlc_labels(points, csv)
