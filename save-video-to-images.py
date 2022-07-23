# Bruno Capuano 2020
# open a video file and save each frame as a PNG file to a folder

import cv2
import os
import numpy as np

video_file = "content/spikeball3.mov"
video_capture = cv2.VideoCapture(video_file)

i = 0
while video_capture.isOpened():

    ret, frameOrig = video_capture.read()
    if ret == True:
        # resize frame, optional you may not need this
        # frame = cv2.resize(frameOrig, frameSize)

        i += 1
        imgNumber = str(i).zfill(5)
        frameImageFileName = str(f'content/spikeball3/image{imgNumber}.png')
        cv2.imwrite(frameImageFileName, frameOrig)
        print(f'{imgNumber} saved')
    else:
        break

video_capture.release()
cv2.destroyAllWindows()