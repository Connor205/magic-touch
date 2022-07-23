import cv2
import numpy as np
import user_input as ui
import math
import tensorflow as tf
import cProfile
import pstats
import logging
from user_dataclasses import Contour, Interval, Possesion
from tqdm import tqdm
from imutils.video import FileVideoStream
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pprint import pprint


class VideoData:
    TIME_BEFORE_CONTACT = 1.0
    TIME_AFTER_CONTACT = 4

    def __init__(self,
                 video_path,
                 nn_path,
                 logger=None,
                 logging_level=logging.INFO):
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - \t%(message)s'))
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        self.logger.setLevel(logging_level)
        self.logger.info(f'{self.__class__.__name__} created')
        self.videoStream = FileVideoStream(video_path, queue_size=512)
        self.videoStream.start()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_shape = (self.frame_height, self.frame_width)
        self.nn = tf.keras.models.load_model(nn_path)
        self.frames_before_contact = self.fps * self.TIME_BEFORE_CONTACT
        self.frames_after_contact = self.fps * self.TIME_AFTER_CONTACT 
        self.ball_bounce_frames = None
        self.point_intervals = None
        self.possesions = None

    def get_ball_bounce_frames(self):
        # This function will return a list of frame numbers where the ball is contained within the selected bounding box
        # The bounding box is defined by the user
        ret, frame = self.cap.read()
        fno = 0
        ball_bounce_frames = []
        user_input = ui.UserInput()
        bounding_box = user_input.selectRectInImage(frame)
        lx = max(bounding_box[0][0] - 25, 0)
        ly = max(bounding_box[0][1] - 25, 0)
        rx = min(bounding_box[1][0] + 25, self.frame_width)
        ry = min(bounding_box[1][1] + 25, self.frame_height)
        w = rx - lx
        h = ry - ly
        self.logger.info(f'Bounding box: ({lx}, {ly}), ({rx}, {ry})')
        fg = cv2.createBackgroundSubtractorMOG2(history=25,
                                                detectShadows=False)
        self.logger.info(f'Created foreground mask object')

        frames_to_skip = 0

        for _ in tqdm(range(self.frame_count)):
            fno += 1
            frame = self.videoStream.read()
            if frame is not None:
                # if  not frames_to_skip == 0:
                #     frames_to_skip = frames_to_skip - 1
                #     continue
                # Lets go ahead and grab the part of the frame that contains the bounding box as defined by the user
                cropped_frame = frame[ly:ry, lx:rx]
                # Now lets check if the ball is contained within the cropped frame
                fgmask = fg.apply(cropped_frame)
                # We can go ahead and perform an erosion on the mask to remove noise
                fgmask = cv2.erode(fgmask, None, iterations=1)
                # Then we will add a box filter to make sure we only find large connected aspects
                fgmask = cv2.boxFilter(fgmask, -1, (5, 5))

                contours, hierarchy = cv2.findContours(fgmask,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                evaluated_contours = []
                for c in contours:
                    circle = cv2.minEnclosingCircle(c)
                    area = cv2.contourArea(c)
                    center = (int(circle[0][0]), int(circle[0][1]))
                    x, y = center
                    radius = int(circle[1])
                    circle_area = math.pi * radius * radius
                    # Lets make sure the image can be grabed due to range limitaitons:
                    if 25 < x and x < w - 25 and 25 < y and y < h - 25:
                        # Then lets make sure it is the right size and is somewhat circular
                        if 150 < area < 1000 and circle_area * .5 < area < circle_area * 1.5:
                            # Then we grab the actual image pixels, these are also changed to RGB instead of BGR
                            image = cropped_frame[np.ix_(
                                range(y - 25, y + 25),
                                range(x - 25, x + 25))][:, :, ::-1]
                            # Now we can run the image through the neural network
                            img_array = tf.expand_dims(
                                image, axis=0)  # Create a batch
                            probs = self.nn(img_array)
                            ss_chance = tf.nn.softmax(probs[0])[1]
                            evaluated_contours.append(
                                Contour(x, y, radius, ss_chance))
                            self.logger.debug("Evaluated a contour")
                if len(evaluated_contours) > 0:
                    best_contour = max(evaluated_contours, key=lambda x: x.ss)
                    if best_contour.ss > .75:
                        ball_bounce_frames.append(fno)
                        self.logger.debug(
                            f"Frame: {fno} - Found spikeball in frame")
            else:
                break
        return ball_bounce_frames

    def get_point_intervals(self) -> list[Interval]:
        if self.ball_bounce_frames is None:
            self.ball_bounce_frames = self.get_ball_bounce_frames()
        
        intervals: list[Interval] = []

        inside = False
        prev = None
        for frame_number in self.ball_bounce_frames:
            if prev is None:
                prev = frame_number
                intervals.append(Interval(start=max(frame_number - self.frames_before_contact, 0), end=None))
                intervals[-1].frames.append(frame_number)
                continue
            if frame_number - prev > self.frames_after_contact:
                intervals[-1].end = prev + self.frames_after_contact
                inside = False
                intervals.append(Interval(start=frame_number - self.frames_before_contact, end=None))
                intervals[-1].frames.append(frame_number)
            else:
                intervals[-1].frames.append(frame_number)
            prev = frame_number
        # Now we should go merge all of the overlapping intervals, since it is possible for some of those to exist
        # The intervals are already sorted by their start time
        merged_intervals: list[Interval] = []
        for interval in intervals:
            if len(merged_intervals) == 0:
                merged_intervals.append(interval)
                continue
            if interval.start <= merged_intervals[-1].end:
                merged_intervals[-1].end = max(interval.end, merged_intervals[-1].end)
                merged_intervals[-1].frames.extend(interval.frames)
            else:
                merged_intervals.append(interval)
        return merged_intervals
    
    def get_possessions(self) -> list[Possesion]:
        if self.point_intervals is None:
            self.point_intervals = self.get_point_intervals()
        possesions: list[Possesion] = []
        for interval in self.point_intervals:
            if interval.end is None:
                continue
            num_net_hits = 0
            frames = interval.frames
            skip_to = None
            for fno in frames:
                if skip_to is None or fno > skip_to:
                    skip_to = fno + self.fps // 3
                    num_net_hits = num_net_hits + 1
            possesions.append(Possesion(num_net_hits, interval.start, interval.end, interval.start / self.fps, interval.end / self.fps, (interval.end - interval.start) / self.fps))
        pprint(possesions)
        return possesions
    
    def cut_video_into_possesions(self, output_path: str, include_faults=False):
        if self.possesions is None:
            self.possesions = self.get_possessions()
        filtered_possesions = self.possesions.copy()
        if not include_faults: 
            filtered_possesions = [x for x in filtered_possesions if x.num_net_hits > 1]
        clips = []
        video = VideoFileClip(self.video_path)
        for possesion in tqdm(filtered_possesions):
            if possesion.end_frame is not None:
                clips.append(video.subclip(possesion.start_time, possesion.end_time))

        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, threads=8, audio=False, fps=self.fps)
        final_clip.close()
        
        
    def cut_video_into_points(self, output_path) -> None:
        if self.point_intervals is None:
            self.point_intervals = self.get_point_intervals()
        clips = []
        video = VideoFileClip(self.video_path)
        for interval in tqdm(self.point_intervals):
            if interval.end is not None:
                clips.append(video.subclip(interval.start//self.fps, interval.end//self.fps))

        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, threads=8, audio=False, fps=self.fps)
        final_clip.close()
        
        