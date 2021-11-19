import cv2
import numpy as np
import time as time 

cap = cv2.VideoCapture("spikeball.mov")
fgbg = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=False)

spikeball_lower_boundary = np.array([0, 0, 0])
spikeball_upper_boundary = np.array([255, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # So we want to go ahead and grab the foreground mask
        foreground = fgbg.apply(frame)
        # Then we can go ahead and grab the pixels that are the correct color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, (30, 30, 25), (35, 255,255))

        final_mask = cv2.bitwise_and(foreground, color_mask)
        filtered_final_mask = cv2.bilateralFilter(final_mask,9,75,75)
        blurred_final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)#cv2.drawContours(frame, filtered_contours, -1, (255,0,0), cv2.FILLED)
        

        # bilateral_image = cv2.bilateralFilter(final_mask, 5, 175, 175)
        # edge_detected_image = cv2.Canny(bilateral_image, 75, 200)
        frame_copy = frame.copy()
        contours, hierarchy = cv2.findContours(filtered_final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            if 150 < cv2.contourArea(contour) < 500:
                #filtered_contours.append(cv2.minEnclosingCircle(contour))
                circle = cv2.minEnclosingCircle(contour)
                if 5 < circle[1] < 15:
                    center = (int(circle[0][0]), int(circle[0][1]))
                    x = center[0]
                    y = center[1]
                    radius = int(circle[1])
                    print(center)
                    print(radius)
                    cv2.circle(frame, center=center, radius=radius, color=(255, 0, 0), thickness=-1)
                    cv2.rectangle(frame, (center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius), (0,255, 0), -1)
                    image = frame_copy[np.ix_(range(y - radius, y + radius),range(x - radius, x + radius))]
                    cv2.imshow("image", image)
        cv2.imshow("actual frame", frame)
        #cv2.imshow("blurred final mask", blurred_final_mask)
        cv2.imshow("filter", filtered_final_mask)
        cv2.imshow("color", color_mask)

        # Press Q on keyboard to  exit
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed
    else:
        break
# When everything done, release 
# the video capture object
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()