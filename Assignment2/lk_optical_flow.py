"""
Author:nz17678
Description: this modules calculates the Lucas-Kanade optical flow for images generated
code adapted from https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
modified to meet the assignment requirement

"""

import cv2
import numpy as np

frame_list = []


def get_lk_optical_flow():
    """
    Description: the fucntion calculates Lucas-Kanade optical flow of the image passed to it. Similar to Dense optical flow
    it first slices the image and extracts frames from it and then calculates the flow.
    here is the example of how it calculates the lucas-kanade flow for the first 100 images from breakout game.
    to do for for more images, simply increase the loop iterations limited by the max number of images available.
    :return:
    """

    # for loop slices the image and extracts frames and appends them to frame_list array

    for i in range(100):
        image = 'data/breakout/images/breakout-' + str(i) + '.png'
        img = cv2.imread(image)

        h = 84
        y = 0
        w = 84
        x = 0

        for j in range(4):
            frame = img[y:h, x:w]
            x += h
            w += h
            frame_list.append(frame)

    colour = np.random.randint(0,255,(100,3))

    # get parameters for Lucase Kanade optical flow operations

    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # get disired parameters, here corners for further processing

    ft_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

    for i in range(len(frame_list)):

        bg_color = cv2.cvtColor(frame_list[i - 1], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(bg_color, mask = None, **ft_params)

        mask = np.zeros_like(frame_list[i - 1])

        frame_color = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(bg_color, frame_color, p0, None, **lk_params)

        # Select good points
        optimal_point_new = p1[st == 1]
        optimal_old_point = p0[st == 1]

        for j, (new, old) in enumerate(zip(optimal_point_new, optimal_old_point)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), colour[j].tolist(), 2)
            frame = cv2.circle(frame_list[j],(a,b),5,colour[j].tolist(),-1)
            output = cv2.add(frame,mask)

        if i == 12003:
            cv2.imwrite('data/breakout/images/LK_optical_flow/breakout_LK_optical_flow.png',output)

            bg_color = frame_color.copy()
            p0 = optimal_point_new.reshape(-1,1,2)


get_lk_optical_flow()
