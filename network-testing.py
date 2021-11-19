import tensorflow as tf
import numpy as np
import cv2
import math
from dataclasses import dataclass
import time


@dataclass
class identified_object:
    x: int
    y: int
    radius: int
    area:int

@dataclass
class circle_to_draw:
    x: int
    y: int
    radius: int

# Lets load the model
model = tf.keras.models.load_model('spikeball-nn.h5')

# Lets load the video

cap = cv2.VideoCapture("spikeball.mov")



fg = cv2.createBackgroundSubtractorMOG2(history=25, detectShadows=False)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("spikeball-with-ball.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (1920, 1080))

ret, frame = cap.read()

fno = 1

ball_list: list[circle_to_draw] = [None] * 30

while ret:
    ret, frame = cap.read()
    if ret:
        prev = time.perf_counter()
        frame_copy = frame.copy()
        # Lets do some processing
        fgmask = fg.apply(frame)
        fgmask = cv2.erode(fgmask, None, iterations=1)
        fgmask = cv2.boxFilter(fgmask, -1, (5, 5))
        # Now lets take that fg mask and grab its contours
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current = time.perf_counter()
        print(f"Took {current - prev} to apply mask and find contours")
        prev = time.perf_counter()
        evaluated_contours = 0
        added_ball = False
        for c in contours:
            circle = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            center = (int(circle[0][0]), int(circle[0][1]))
            x = center[0]
            y = center[1]
            radius = int(circle[1])
            circle_area = math.pi * radius * radius
            # lets make sure the image can be grabed due to range limitaitons:
            if 25 < x and x < 1895 and 25 < y and y < 1055:
                # Now lets check to see if the radius meets spec
                # if not 2 < radius < 20:
                #     cv2.circle(frame, center=center, radius=radius, color=(255, 0, 0), thickness=3)
                # if not area * .5 < circle_area < area * 1.5:
                #     cv2.circle(frame, center=center, radius=radius, color=(255, 255, 255), thickness=3)
                # if not 150 < area < 1000 and not circle_area * .5 < area < circle_area * 1.5:
                #     cv2.circle(frame, center=center, radius=radius, color=(255, 255, 255), thickness=3)
                if 150 < area < 1000 and circle_area * .5 < area < circle_area * 1.5:
                    evaluated_contours += 1
                    #item = identified_object(x, y, radius, area)
                    # Now lets grab the image
                    image = frame_copy[np.ix_(range(y - 25, y + 25),range(x - 25, x + 25))][:,:,::-1]
                    # Now lets pipe it into the network
                    img_array = tf.expand_dims(image, axis=0) # Create a batch
                    nn_start = time.perf_counter()
                    y_proba = model(img_array)
                    # y_proba = model.predict(img_array)
                    nn_end = time.perf_counter()
                    #print(f"Took {nn_end - nn_start} to run the network")
                    score = tf.nn.softmax(y_proba[0])
                    #print(score)
                    if score[1] > 0.8:
                        #cv2.circle(frame, center=center, radius=radius, color=(0, 255, 0), thickness=-1)
                        ball_list.append(circle_to_draw(x, y, radius))
                        added_ball = True
                        break
                    # else:
                    #     cv2.circle(frame, center=center, radius=radius, color=(0, 0, 255), thickness=3)
        if not added_ball:
            ball_list.append(None)
        for ball in ball_list[:-1]:
            if ball is not None:
                #print(ball.x, ball.y, ball.radius)
                cv2.circle(frame, center=(ball.x, ball.y), radius=(ball.radius  // 2), color=(0, 255, 0), thickness=-1)
        current = time.perf_counter()
        print(f"Took {current - prev} to iterate over {len(contours)} contours and eval {evaluated_contours} of them")
        writer.write(frame)
        print(f"Wrote frame {fno}")
        fno += 1
        ball_list.pop(0)

writer.release()
            



