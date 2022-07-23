import cv2 as cv2
import numpy as np
import math
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
from user_dataclasses import Interval
from video_data import VideoData

def closeWindows():
    for _ in range(10):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def saveVideoClips(video_path: str, intervals: list[Interval], output_path: str):
    clips = []
    video = VideoFileClip(video_path)
    for interval in tqdm(intervals):
        if interval.end is not None:
            clips.append(video.subclip(interval.start//60, interval.end//60))

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, threads=8, audio=False, fps=60)
    final_clip.close()