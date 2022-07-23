import cv2
import os
import numpy as np
import shutil

directory = "content/spikeball3"

files = os.listdir(directory)
files.sort()

for f in files:
    cv2.imshow("Output", cv2.imread(f"{directory}/{f}"))
    cv2.waitKey(1)
