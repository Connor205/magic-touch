# This program will trace the ball and draw annotate the balls path over the entire video
import argparse
from typing import Counter
import cv2
import numpy as np
import tensorflow as tf
import logging
from tqdm import tqdm
from dataclasses import dataclass
import math
import time
import pickle
import cProfile
import pstats
from dataclasses import Position, FrameData, Contour


class MainClass:
    def __init__(self, args) -> None:
        logger.debug("Running init")
        logger.debug("args: %s", args)
        self.filename = args.filename
        self.color = args.color
        self.nn = tf.keras.models.load_model(args.nn)
        logger.debug("Successfully loaded neural network")
        logger.debug(f"NN: {self.nn}")
        self.cap = cv2.VideoCapture(self.filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug("Successfully loaded video")
        self.output_filename = args.output
        self.frames = args.frames
        self.display = args.display

    def run(self) -> None:
        # Lets set up some initial cv2 stuff
        fg = cv2.createBackgroundSubtractorMOG2(history=25,
                                                detectShadows=False)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(self.output_filename, fourcc, self.fps,
                                 (1920, 1080))
        past_positions: list[Position] = [None] * 30
        frame_data: list[FrameData] = []
        # Lets go ahead and iterate through all of the frames
        for fno in tqdm(range(self.frame_count)):
            read_start = time.perf_counter()
            ret, frame = self.cap.read()
            read_end = time.perf_counter()
            read_time = read_end - read_start
            logger.debug(f"Frame {fno}: read in frame in {read_time} seconds")
            if not ret:
                break
            # Check to see if we want to process this frame
            if fno % self.frames == 0:
                mask_start = time.perf_counter()
                # Lets do some processing
                # So the first thing we want to do is generate the foreground mask
                fgmask = fg.apply(frame)
                # We can go ahead and perform an erosion on the mask to remove noise
                fgmask = cv2.erode(fgmask, None, iterations=1)
                # Then we will add a box filter to make sure we only find large connected aspects
                fgmask = cv2.boxFilter(fgmask, -1, (5, 5))
                mask_end = time.perf_counter()
                mask_time = mask_end - mask_start
                logger.debug(
                    f"Frame {fno}: Took {mask_time} seconds to generate mask")

                contour_find_start = time.perf_counter()
                # Now lets take that fg mask and grab its contours, specifically we only want the outer contours
                contours, hierarchy = cv2.findContours(fgmask,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                contour_find_end = time.perf_counter()
                contour_find_time = contour_find_end - contour_find_start
                logger.debug(
                    f"Frame {fno}: Took {contour_find_time} seconds to find contours"
                )
                contour_start = time.perf_counter()
                # Then we can iterage over all the contours and evaluate them
                evaluated_contours: list[Contour] = []
                for c in contours:
                    circle = cv2.minEnclosingCircle(c)
                    area = cv2.contourArea(c)
                    center = (int(circle[0][0]), int(circle[0][1]))
                    x, y = center
                    radius = int(circle[1])
                    circle_area = math.pi * radius * radius
                    # Lets make sure the image can be grabed due to range limitaitons:
                    if 25 < x and x < 1895 and 25 < y and y < 1055:
                        # Then lets make sure it is the right size and is somewhat circular
                        if 150 < area < 1000 and circle_area * .5 < area < circle_area * 1.5:
                            # Then we grab the actual image pixels, these are also changed to RGB instead of BGR
                            image = frame[np.ix_(range(y - 25, y + 25),
                                                 range(x - 25,
                                                       x + 25))][:, :, ::-1]
                            # Now we can run the image through the neural network
                            img_array = tf.expand_dims(
                                image, axis=0)  # Create a batch
                            probs = self.nn(img_array)
                            ss_chance = tf.nn.softmax(probs[0])[1]
                            evaluated_contours.append(
                                Contour(x, y, radius, ss_chance))
                logger.debug("Evaluated contours: %d", len(evaluated_contours))
                if evaluated_contours != []:
                    # Now we can sort the contours by their ss chance
                    evaluated_contours.sort(key=lambda x: x.ss)
                    highest_ss = evaluated_contours[-1]
                    # check to see if the highest ss is greater than .75, if not skip it
                    if highest_ss.ss > .75:
                        past_positions.append(highest_ss)
                    else:
                        past_positions.append(None)
                else:
                    past_positions.append(None)
                contour_end = time.perf_counter()
                contour_time = contour_end - contour_start
                logger.debug(
                    f"Frame {fno}: Took {contour_time} seconds to evaluate contours"
                )

                frame_data.append(
                    FrameData(
                        fno, True, read_time, mask_time, contour_find_time,
                        len(contours), len(evaluated_contours), contour_time,
                        contour_time / len(evaluated_contours) if
                        evaluated_contours != [] else -1, past_positions[-1]))
            else:
                # If we are not processing this frame we should add a blank to the list
                past_positions.append(None)
                frame_data.append(FrameData(fno, False))

            # So regardless of whether we are processing or not, we want to draw the past circles
            for i, p in enumerate(past_positions[:-1]):
                if p is not None:
                    cv2.circle(frame, (p.x, p.y), p.radius // 2, (0, 255, 0),
                               -1)

            # Then we pop the oldest one
            past_positions.pop(0)
            write_start = time.perf_counter()
            # Now that the frame is fully complete we can write it to the output file
            writer.write(frame)
            write_end = time.perf_counter()
            logger.debug(
                f"Frame {fno}: Took {write_end - write_start} seconds to write frame"
            )
            # And we can show the frame
            if self.display:
                cv2.imshow("Output", frame)
                cv2.waitKey(1)
        pickle.dump(frame_data,
                    open(f"{self.output_filename}_framedata.p", "wb"))
        cv2.destroyAllWindows()
        self.cap.release()


if __name__ == "__main__":
    # Parse arguments for filename and color
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename",
                        "-f",
                        required=True,
                        help="The filename of the video to process")
    parser.add_argument("--color",
                        "-c",
                        help="The color to trace the ball with")
    parser.add_argument("--debug",
                        "-d",
                        action="store_true",
                        help="Enable debug output")
    parser.add_argument("--nn",
                        "-n",
                        required=True,
                        help="The path to the neural network to use")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display output to screen frame by frame as each frame is loaded")
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help=
        "Only frame numbers divisible by this number will be processed (other frames will not be analyzed)"
    )
    parser.add_argument("--output",
                        "-o",
                        required=True,
                        help="The path to the output file")
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    main = MainClass(args)
    with cProfile.Profile() as pr:
        main.run()
    stats = pstats.Stats(pr)
    stats = stats.sort_stats('cumulative')
    stats.print_stats()
    stats.dump_stats(f"{args.output}_profile.txt")
