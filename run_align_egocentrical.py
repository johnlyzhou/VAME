import vame
import os

# These paths have to be set manually
working_directory = '/Users/johnzhou/code'
project = 'ps_vae'
videos = ['/Users/johnzhou/code/dlc_videos/B29_post_sideDLC_resnet101_FearConditioningTopNov27shuffle1_950000_filtered.mp4']
csv = '/Users/johnzhou/code/dlc_videos/B29_post_sideDLC_resnet101_FearConditioningTopNov27shuffle1_950000_filtered.csv'

# Initialize your project
config = vame.init_new_project(
    project=project, videos=videos, working_directory=working_directory, videotype='.mp4')
os.system('cp /Users/johnzhou/code/dlc_videos/B29_post_sideDLC_resnet101_FearConditioningTopNov27shuffle1_950000_filtered.csv /Users/johnzhou/code/ps_vae-Jul19-2021/videos/pose_estimation')

# Run egocentric alignment and save video
vame.egocentric_alignment(
    config, pose_ref_index=[0, 4], use_video=True, check_video=True, save_video=True)
os.system('rm -rf /Users/johnzhou/code/ps_vae-Jul19-2021')
