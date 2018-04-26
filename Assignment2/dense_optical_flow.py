"""
Author:nz17678
Description: this modules calculates the dense optical flow for images generated
code adapted from https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
modified to meet the assignment requirement
"""
import cv2
import numpy as np

frame_list = []
optical_flow = []


def get_dense_optical_flow():
    """
    the function calculates dense optical flow between the first and last last frame of four-frame concatenated images
    first it breaks the images into four slices and then feeds the sliced array into the openCV repsective function to
    calculate the flow
    :return:
    """

    for i in range(3001):
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

    l = len(frame_list)

    for i in range(l):
        previous = cv2.cvtColor(frame_list[i - 1], cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame_list[i])
        hsv[...,1] = 255
        next = cv2.cvtColor(frame_list[i],cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous,next, None, 0.5, 1, 30, 3, 7, 1.5, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        optical_flow.append(flow[0])

        if i == l - 1:
            cv2.imwrite('data/breakout/images/dense_optical_flow/breakout_flow_'+ str(i) + '.png',bgr)

        previous = next


get_dense_optical_flow()
