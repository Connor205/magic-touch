import cv2
import numpy as np
from user_dataclasses import *
import logging
from util_functions import *


class UserInput:
    def __init__(self, logger=None, logging_level=logging.INFO):
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

    def getClicks(self, img, window_name="Get User Points", num_clicks=1):
        '''
        Get user clicks on an image.
        :param img: image to get clicks on
        :param num_clicks: number of clicks to get
        :return: list of points
        '''
        points = []
        collected_points = False

        def record_click(event, x, y, flags, param):
            if len(points) < num_clicks:
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    self.logger.debug(
                        f'Point {len(points)} recorded: {x}, {y}')

        cv2.namedWindow(window_name)

        cv2.setMouseCallback(window_name, record_click)
        while not collected_points:
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
            if len(points) == num_clicks:
                collected_points = True
        closeWindows()
        return points

    def selectRectInImage(self, img):
        '''
        Select a rectangle in an image.
        :param img: image to select rectangle in
        :return: tuple of points
        '''
        self.logger.info(
            'Please select the upper left and bottom right corner of the spikeball net bounding box'
        )

        points = self.getClicks(img, num_clicks=2)
        self.logger.info(f'Points selected: {points}')
        self.logger.info("The selected rectangle is now shown on the display")
        img_copy = img.copy()
        cv2.rectangle(img_copy, points[0], points[1], (0, 0, 255), 5)
        self.logger.info(
            "If you would like to redo the points then please press r, otherwise click any other key"
        )
        cv2.imshow('Selected Rectangle', img_copy)
        k = cv2.waitKey(0)
        if k == ord('r'):
            self.logger.info("Redoing points")
            points = self.selectRectInImage(img)
        closeWindows()
        return points