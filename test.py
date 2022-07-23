import cv2
from user_input import UserInput
from video_data import VideoData
import cProfile
import pstats

vd = VideoData("content/spikeball3.mov", "spikeball-nn.h5")

with cProfile.Profile() as pr:
    vd.cut_video_into_possesions("spikeball3_cut.mp4")
stats = pstats.Stats(pr)
stats = stats.sort_stats('cumulative')
stats.print_stats()
stats.dump_stats(f"spikeball3_profile.txt")
