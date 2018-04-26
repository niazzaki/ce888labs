"""
Author:nz17678
Description: this modules module explains the
code partially adapted from https://github.com/marcotcr/lime and other helpful sources on Github.
modified to meet the assignment requirement

"""

from keras.preprocessing import image
import lime
from lime import lime_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from common_ass2 import play_one_episode

# get a list of actions from the acts file

action_data = 'data/breakout/acts/breakout.txt'
f = open(action_data)

# set the model

model = play_one_episode

#load the environment

environment = "/tensorpack/Examples/A3C-Gym/Breakout-v0.npz"
output = []
actset = []


def get_prediction(s, t):

    return model(s, t)


def explain_with_lime():
    for i in range(50):
        img = cv2.imread('data/breakout/images/breakout-' + str(i) + '.png')

        # Revert image to array
        for it in range(4):
            img = image.img_to_array(img[:, :, it * 3:3 * (it + 1)])

            output.append(img)

        actions = str(f.readline())
        actset.append(actions)


        predictions = get_prediction(output[i][i][i][0], actset[i])


    explainer = lime_image.LimeImageExplainer()

    # Hide color is the color for a superpixel turned OFF. Alternatively, if
    # it is NONE, the superpixel will be replaced by the average of its pixels

    explanation = explainer.explain_instance(output[0][0], actset[0], model, hide_color=0, num_samples=1000)

    from skimage.segmentation import mark_boundaries

    temp, mask = explanation.get_image_and_mask(295, positive_only=True, num_features=5, hide_rest=True)
    fig = (mark_boundaries(temp / 2 + 0.5, mask))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    fig.savefig("data/breakout/LIME_output/LIME_breakout.png")


explain_with_lime()
