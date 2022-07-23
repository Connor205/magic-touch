# Connor Nelson
# Edited 5/23/2022

# Grabbing user inputs for directories of content
print("Loading In Libraries")
from json import encoder
import pathlib
import logging
import cv2
from pprint import pprint
import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips
from imutils.video import FileVideoStream as Fvs
import dataclasses
import user_input
from user_dataclasses import Contour, Interval, Possesion

print("Finished Loading Libraries")

ui = user_input.UserInput()
TIME_BEFORE_CONTACT = 1.0
TIME_BEFORE_FIRST_CONTACT = 0.5
TIME_AFTER_CONTACT = 4
TIME_AFTER_LAST_CONTACT = 2.0


def getNetCoordsMan(fileName):
    vid = cv2.VideoCapture(fileName)
    success, image = vid.read()

    if not success:
        raise Exception("Could not read video file")
    vid.release()
    bounding_box = ui.selectRectInImage(image)
    lx = max(bounding_box[0][0] - 25, 0)
    ly = max(bounding_box[0][1] - 25, 0)
    rx = min(bounding_box[1][0] + 25, image.shape[1])
    ry = min(bounding_box[1][1] + 25, image.shape[0])
    return [ly, lx, ry, rx]


class Main:
    def __init__(self, nnPath):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.nn = tf.keras.models.load_model(nnPath)

    def getFiles(self):
        # Gets user input, checks if it is a directory or file, and then creates paths list
        self.input_path = input("Path: ")
        self.input_path = self.input_path.strip()
        self.input_path = pathlib.Path(self.input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"{self.input_path} does not exist.")

        self.input_path.resolve()
        if self.input_path.is_dir():
            self.logger.debug("Directory selected")
            self.paths = list(self.input_path.iterdir())
        elif self.input_path.is_file():
            self.logger.debug("File selected")
            self.paths = [self.input_path]
        newPaths = []
        for p in self.paths:
            if p.suffix.lower() == ".mp4" or p.suffix.lower() == ".mov":
                newPaths.append(p)
        self.paths = newPaths
        if len(self.paths) == 0:
            raise ValueError("No video files found")
        self.logger.debug(self.paths)

    def generateVideoDictionary(self):
        # Generates a dictionary of video names associated stats and their associated paths
        self.videos = {}
        for p in self.paths:
            self.videos[p.stem] = {}
            self.videos[p.stem]["path"] = str(p)
            vid = cv2.VideoCapture(str(p))
            self.videos[p.stem]["fps"] = vid.get(cv2.CAP_PROP_FPS)
            self.videos[p.stem]['fpsRounded'] = round(
                self.videos[p.stem]['fps'])
            self.videos[p.stem]["numFrames"] = vid.get(
                cv2.CAP_PROP_FRAME_COUNT)
            self.videos[p.stem]['duration'] = self.videos[
                p.stem]['numFrames'] / self.videos[p.stem]['fps']
            self.videos[p.stem]['corners'] = getNetCoordsMan(str(p))
            vid.release()

        pprint(self.videos)

    def findKeyFrames(self):
        # Finds the key frames for each video
        for key, data in self.videos.items():
            self.logger.debug("Finding key frames for " + key)
            fileStream = Fvs(data['path']).start()
            vid = cv2.VideoCapture(data["path"], cv2.CAP_FFMPEG)
            ly, lx, ry, rx = data['corners']
            w = rx - lx
            h = ry - ly
            success, frame = vid.read()
            keyFrames = []
            fno = 1
            fg = cv2.createBackgroundSubtractorMOG2(history=25,
                                                    detectShadows=False)
            if not success:
                raise Exception("Could not read video file")
            for fno in tqdm(range(1, int(data['numFrames']) + 1)):
                frame = fileStream.read()
                fno = fno + 1
                if not success or frame is None:
                    break
                cropped_frame = frame[ly:ry, lx:rx]
                fgmask = fg.apply(cropped_frame)
                fgmask = cv2.erode(fgmask, None, iterations=1)
                fgmask = cv2.boxFilter(fgmask, -1, (5, 5))
                # Lets only evaluate contours 15 times per second, this will speed up higher fps videos
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
                            # self.logger.debug("Evaluated a contour")
                if len(evaluated_contours) > 0:
                    best_contour = max(evaluated_contours, key=lambda x: x.ss)
                    if best_contour.ss > .75:
                        keyFrames.append(fno)
                        # self.logger.debug(
                        #     f"Frame: {fno} - Found spikeball within net boundaries"
                        # )
            self.videos[key]['keyFrames'] = keyFrames
            vid.release()

    def createIntervals(self):
        for key, data in self.videos.items():
            frames_before = int(data['fps'] * TIME_BEFORE_CONTACT)
            frames_after = int(data['fps'] * TIME_AFTER_CONTACT)
            frames_after_last_contact = int(data['fps'] *
                                            TIME_AFTER_LAST_CONTACT)
            frames_before_first_contact = int(data['fps'] *
                                              TIME_BEFORE_FIRST_CONTACT)
            intervals: list[Interval] = []
            prev = None
            for frame_number in data['keyFrames']:
                if prev is None:
                    prev = frame_number
                    intervals.append(
                        Interval(start=max(
                            frame_number - frames_before_first_contact, 0),
                                 end=None))
                    intervals[-1].frames.append(frame_number)
                    continue
                if frame_number - prev > frames_after:
                    intervals[-1].end = prev + frames_after_last_contact
                    intervals.append(
                        Interval(start=frame_number - frames_before, end=None))
                    intervals[-1].frames.append(frame_number)
                else:
                    intervals[-1].frames.append(frame_number)
                prev = frame_number
            # Taking care of a final interval that extends path the end of the video
            if intervals[-1].end is None:
                intervals[-1].end = frame_number
            # Now we should go merge all of the overlapping intervals, since it is possible for some of those to exist
            # The intervals are already sorted by their start time
            merged_intervals: list[Interval] = []
            for interval in intervals:
                if len(merged_intervals) == 0:
                    merged_intervals.append(interval)
                    continue
                if interval.start <= merged_intervals[-1].end:
                    merged_intervals[-1].end = max(interval.end,
                                                   merged_intervals[-1].end)
                    merged_intervals[-1].frames.extend(interval.frames)
                else:
                    merged_intervals.append(interval)
            self.videos[key]['intervals'] = merged_intervals

    def createPossessions(self):
        for key, data in self.videos.items():
            possesions: list[Possesion] = []
            for interval in data['intervals']:
                if interval.end is None:
                    continue
                num_net_hits = 0
                frames = interval.frames
                skip_to = None
                for fno in frames:
                    if skip_to is None or fno > skip_to:
                        skip_to = fno + int(data['fps']) // 3
                        num_net_hits = num_net_hits + 1
                possesions.append(
                    Possesion(num_net_hits, interval.start, interval.end,
                              interval.start / data['fps'],
                              interval.end / data['fps'],
                              (interval.end - interval.start) / data['fps']))
            self.videos[key]['possesions'] = possesions

    def createPointVideo(self, outputPath, min_possesions=2):
        self.logger.debug("Creating Point Video")
        concatedVideos = []
        fullVideos = []
        for key, data in self.videos.items():
            self.logger.debug(f"Creating Clip For Video: {key}")
            clipList = []
            fps = data['fps']
            self.logger.debug(
                f"This Video has {len(data['possesions'])} possesions")

            fullVideo = VideoFileClip(str(data['path']))
            fullVideos.append(fullVideo)
            for possesion in data['possesions']:
                if possesion.num_net_hits < min_possesions:
                    continue

                clip = fullVideo.subclip(possesion.start_time,
                                         possesion.end_time)
                clipList.append(clip)
            concatedVideos.append(
                concatenate_videoclips(clipList, method="chain"))
        concatenate_videoclips(concatedVideos).write_videofile(outputPath,
                                                               threads=8)
        for fullVideo in fullVideos:
            fullVideo.close()

    def writePosessionsToFile(self):
        class EnhancedJSONEncoder(json.JSONEncoder):
            def default(self, o):
                if dataclasses.is_dataclass(o):
                    return dataclasses.asdict(o)
                return super().default(o)

        for key, data in self.videos.items():
            with open(f"{key}-analysis.json", 'w') as file:
                # Write the data from possesion into a json file
                json.dump(data, file, indent=4, cls=EnhancedJSONEncoder)

    def main(self):
        print(
            "Hello! Welcome to Magic Touch!\nI am a command line interface that will help you generate highlight clips for your spikeball footage.\nPlease provide the path to the directory or video file you would like to analyze."
        )
        self.getFiles()
        self.generateVideoDictionary()
        self.findKeyFrames()
        self.createIntervals()
        self.createPossessions()
        # pprint(self.videos)
        self.writePosessionsToFile()

        # self.createPointVideo('test_clippingB.mp4')


if __name__ == "__main__":
    main = Main("spikeball-nn.h5")
    main.main()
