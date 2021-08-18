import numpy as np
import pandas as pd
import tqdm

points = np.load('dlc_points.npy')

data = pd.read_csv('/Users/johnzhou/code/dlc_videos'
                   '/B125_precon_topDLC_resnet101_FearConditioningTopNov27shuffle1_950000_filtered.csv', skiprows=2)
csv = pd.read_csv('/Users/johnzhou/code/dlc_videos'
                  '/B125_precon_topDLC_resnet101_FearConditioningTopNov27shuffle1_950000_filtered.csv',
                  low_memory=False)

frames, body_parts, coords = points.shape

for f in tqdm.tqdm(range(frames), disable=not True, desc='Writing points'):
    for b in range(body_parts):
        csv.iloc[f + 2, b * 3 + 1] = points[f, b, 0]
        csv.iloc[f + 2, b * 3 + 2] = points[f, b, 1]

csv.to_csv('shifted_dlc.csv', index=False)
