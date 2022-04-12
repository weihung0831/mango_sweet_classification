import pretty_errors
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


def read_rgb_image_path(filename):
    read_rgb_image = cv.imread(filename)
    resize_rgb_image = cv.resize(read_rgb_image, (224, 224), cv.INTER_CUBIC)

    print(filename)
    print(read_rgb_image.shape)
    print(resize_rgb_image.shape)

    for i in range(184):
        cv.imshow('Resize rgb Image', resize_rgb_image)
        cv.waitKey(1)

    return resize_rgb_image


def read_depth_image_path(filename):
    read_depth_image = cv.imread(filename, 0)
    resize_depth_image = cv.resize(read_depth_image, (224, 224), cv.INTER_CUBIC)

    print(filename)
    print(read_depth_image.shape)
    print(resize_depth_image.shape)

    for i in range(184):
        cv.imshow('Resize depth Image', resize_depth_image)
        cv.waitKey(1)

    return resize_depth_image


def combine_image(resize_rgb, resize_depth):
    b, g, r = cv.split(resize_rgb)
    rgbd_image = cv.merge([r, g, b, resize_depth])

    return rgbd_image


def load_rgbd_image(filename):
    # rgbd_image = np.load(filename)['rgbd'] #dict
    load_rgbd_image = np.load(filename)

    print(filename)
    print(load_rgbd_image['rgbd'].shape)

    return load_rgbd_image


if __name__ == "__main__":
    # rgb_path = 'dataset/100_100/rgb'
    # depth_path = 'dataset/100_100/depth'
    # rgbd_path = 'new_data/rgbd'
    #
    # for filename in os.listdir(rgb_path):
    #     rgb_file = os.path.join(rgb_path, filename)
    #     depth_file = os.path.join(depth_path, filename.replace('Color', 'Depth'))
    #
    #     resize_rgb_image = read_rgb_image_path(rgb_file)
    #     resize_depth_image = read_depth_image_path(depth_file)
    #     rgbd_image = combine_image(resize_rgb_image, resize_depth_image)
    #
    #     rgbd_file = os.path.join(rgbd_path, filename.replace('Color.png', 'RGBD.sweet_npz'))
    #     np.savez(rgbd_file, rgbd=rgbd_image, rgb=resize_rgb_image, depth=resize_depth_image)
    #
    # for filename in os.listdir(rgbd_path):
    #     rgbd_file = os.path.join(rgbd_path, filename)
    #     rgbd_image = load_rgbd_image(rgbd_file)

    npz_path = './dataset/100_100/npz/0.npz'
    npz_info = load_rgbd_image(npz_path)

    # print('rgb 照片數量：', len(os.listdir('dataset/400_300/sweet/rgb')), '張')
    # print('depth 照片數量：', len(os.listdir('dataset/400_300/sweet/depth')), '張')
    # print('RGBD 照片數量：', len(os.listdir('dataset/400_300/sweet/rgbd')), '張')
