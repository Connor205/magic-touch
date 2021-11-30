import cv2
import numpy as np
import user_input as ui
import math
import tensorflow as tf
import cProfile
import pstats
import logging
from user_dataclasses import Contour
from tqdm import tqdm
from imutils.video import FileVideoStream


class VideoData:
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
        self.logger.info(f'Bounding box: {lx}, {ly}, {rx}, {ry}')
        fg = cv2.createBackgroundSubtractorMOG2(history=25,
                                                detectShadows=False)
        self.logger.info(f'Created foreground mask')

        for _ in tqdm(range(self.frame_count)):
            fno += 1
            frame = self.videoStream.read()
            if frame is not None:
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
                # cv2.imshow('fgmask', fgmask)
                # cv2.imshow('cropped_frame', cropped_frame)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(1)
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
