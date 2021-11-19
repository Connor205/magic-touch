import cv2 
import numpy as np
import math

# So first we want to go ahead and generate all of the images
# This is done by going through all of the suspected balls frame by frame
# We do not care about the accuracy of these steps beyond the fact that we 
# want very few false negatives
# If the ball is being tracked as one of the objects thats what we care about

cap = cv2.VideoCapture("spikeball5.mov")

# We can go ahead and define some constants 
fgbgMOG2 = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)
fgbgKNN = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=False)
spikeball_lower_color = (10, 35, 60)
spikeball_upper_color = (100, 255,200)
black_lower = (0, 34, 0)
black_upper = (73, 248, 85)
count = 0
finished = False
frame_count = 0

while cap.isOpened() and not finished:
    # for i in range(10):
    #     ret, frame = cap.read()
    ret, frame = cap.read()
    frame_count += 1
    if ret:
        frame_copy = frame.copy()
        hsv_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # So first we are going to grab the foreground mask
        fgmask = fgbgMOG2.apply(frame)
        fgmask = cv2.erode(fgmask, None, iterations=1)
        fgmask = cv2.dilate(fgmask, None, iterations=1)
        # fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        yellow_mask = cv2.inRange(hsv_frame, spikeball_lower_color, spikeball_upper_color)
        black_mask = cv2.inRange(hsv_frame, black_lower, black_upper)
        colormask = cv2.bitwise_or(yellow_mask, black_mask)
        colormask = cv2.dilate(colormask, None, iterations=2)
        finalmask = cv2.bitwise_and(fgmask, colormask)
        finalmask = fgmask

        # Now we can find the contours of the ball
        contours, hierarchy = cv2.findContours(finalmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        balls_per_frame = 0
        for c in contours:
            area = cv2.contourArea(c)
            circle = cv2.minEnclosingCircle(c)
            center = (int(circle[0][0]), int(circle[0][1]))
            x = center[0]
            y = center[1]
            radius = int(circle[1])
            circle_area = math.pi * radius * radius
            # We are going to check to see if the countor is circle like
        
            if radius > 3:
                if circle_area * 0.8 < area < circle_area * 1.2:
                    #print("Found Contour Fitting Description")
                    # Then we are going to check to make sure the circle is in bounds
                    if x - 25 > 0 and x + 25 < 1920 and y - 25 > 0 and y + 25 < 1080:
                        count += 1
                        # Then we can take that contour and draw it for reference on the frame
                        cv2.drawContours(frame, [c], -1, (255, 0, 0), -1)
                        b_image = frame_copy[np.ix_(range(y - 25, y + 25),range(x - 25, x + 25))]
                        cv2.imwrite("spikeball_5_" + str(count) + ".jpg", b_image)
                        balls_per_frame += 1
                # else:
                #     if x - 25 > 0 and x + 25 < 1980 and y - 25 > 0 and y + 25 < 1080:
                #         # Then we can take that contour and draw it for reference on the frame
                #         cv2.drawContours(frame, [c], -1, (0, 0, 255), -1)
        print(f"Frame {frame_count}: {balls_per_frame} balls")
        print(f"Total: {count}")


        # balls = [cv2.minEnclosingCircle(cnt) for cnt in contours if 150 < cv2.contourArea(cnt) < 300]
        
        # for b in balls:
        #     center = (int(b[0][0]), int(b[0][1]))
        #     x = center[0]
        #     y = center[1]
        #     radius = int(b[1])
        #     if 5 < radius < 30 and x - 25 > 0 and x + 25 < 1920 and y - 25 > 0 and y + 25 < 1080:
        #         b_image = frame_copy[np.ix_(range(y - 25, y + 25),range(x - 25, x + 25))]
        #         # cv2.imshow("ball image", cv2.resize(b_image, (500, 500)))
        #         # if cv2.waitKey(25) & 0xFF == ord('q'):
        #         #     break
        #         cv2.circle(frame, center=center, radius=radius, color=(255, 0, 0), thickness=3)
        #         if count < 10000:
        #             #cv2.imwrite("spikeball_1_" + str(count) + ".jpg", b_image)
        #             count += 1
        #             print(count)
        #         else:
        #             print("WARNING: FINISHED")
        #             finished = True
        # cv2.imshow("fgmask", fgmask)
        # cv2.imshow("colormask", colormask)
        # cv2.imshow("finalmask", finalmask)
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    else:
        break


cap.release()
cv2.destroyAllWindows()
