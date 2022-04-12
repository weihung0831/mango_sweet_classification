import pretty_errors
import cv2 as cv
import matplotlib.pyplot as plt
import os


def cut_image(src):
    rgb_img = cv.imread(src)
    cut_img = rgb_img[60:500, 230:1100]

    print(src)

    for i in range(244):
        cv.imshow('Cut Image', cut_img)
        cv.waitKey(1)

    return cut_img


if __name__ == '__main__':
    rgb_path = 'image'
    cut_path = 'result'

    for filename in os.listdir(rgb_path):
        rgb_file = os.path.join(rgb_path, filename)
        cut_img = cut_image(rgb_file)
        # cut_image = cv.cvtColor(cut_image, cv.COLOR_BGR2RGB)

        # plt.imshow(cut_image)
        # plt.show()
        rgbd_file = os.path.join(cut_path, filename)
        cv.imwrite('result/{}'.format(filename), cut_img)
