import vame
import os
import datetime

# These paths have to be set manually
working_directory = '/Users/johnzhou/code'
project = 'ps_vae'
videos = ['/Users/johnzhou/code/VAME/results/310_480_masked.mp4']
csv = '/Users/johnzhou/code/VAME/results/310_480_masked.csv'

mydate = datetime.datetime.now()
short_month = mydate.strftime("%b")
date = mydate.date().strftime("%-d")

# Initialize your project
config = vame.init_new_project(
    project=project, videos=videos, working_directory=working_directory, videotype='.mp4')
os.system('cp {} {}/ps_vae-{}{}-2021/videos/pose_estimation'.format(working_directory, csv, short_month, date))

# Run egocentric alignment and save video
vame.egocentric_alignment(
    config, pose_ref_index=[0, 3], crop_size=(192, 192), use_video=True, check_video=False, save_video=True)
os.system('rm -rf {}/ps_vae-{}{}-2021'.format(working_directory, short_month, date))
