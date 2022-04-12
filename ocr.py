import pretty_errors
import glob
import re
import PIL
import cv2 as cv
import imutils
import os
from pytesseract import image_to_string
import matplotlib.pyplot as plt


# 獲取資料夾裡所有png
def get_files(directory, filetype='.png'):
    filelist = []
    for file_ in os.listdir(directory):
        if file_.endswith(filetype):
            filelist.append(os.path.join(directory, file_))

    return filelist


def show_rgb(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.subplot(221)
    plt.title('rgb Image')
    plt.imshow(img)
    # plt.show()


def show_gray(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.subplot(222)
    plt.title('GRAY Image')
    plt.imshow(img, cmap='gray')
    # plt.show()


# 提高對比度
def clache(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clache = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clache.apply(img)


def noise(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.copy()
    img = img[530:630, 180:600]  # 切割圖片(需要調整部分)
    plt.subplot(223)
    plt.title('Gray Copy Image')
    plt.imshow(img, cmap='gray')
    # plt.show()


def remove_noise(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.copy()
    img = img[530:630, 180:600]  # 切割圖片(需要調整部分)
    img = cv.bilateralFilter(img, 5, 7, 7)

    plt.subplot(224)
    plt.title('Remove Noise Image')
    plt.imshow(img, cmap='gray')
    plt.show()

    return img


def clean_display(img_arr):
    image = imutils.resize(img_arr, height=300)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = cv.bilateralFilter(gray_image, 17, 11, 11)
    edged_image = cv.threshold(gray_image, 30, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # clache = cv.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    # cl1 = clache.apply(gray_image)
    # gray_image = cv.bilateralFilter(cl1, 17, 11, 11)
    # edged_image = cv.threshold(gray_image, 30, 256, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    for i in range(3):
        edged_image = cv.erode(edged_image, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 11)), 30)
        edged_image = cv.dilate(edged_image, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), 5)

    edged_image[0:22, 0:-1] = 255

    # cv.imwrite('cntr_ocr.jpg', edged_image)
    return edged_image


def process_image(orig_image_arr):
    ratio = orig_image_arr.shape[0] / 300.0
    display_image_arr = clean_display(orig_image_arr)

    return display_image_arr


# def process_image(orig_image_arr):
#     ratio = orig_image_arr.shape[0] / 300.0
#     display_image_arr = clean_display(orig_image_arr)
#
#     return display_image_arr


def ocr_image(orig_image_arr):
    otsu_thresh_image = PIL.Image.fromarray(process_image(orig_image_arr))
    string = image_to_string(otsu_thresh_image, lang="lets", config="--psm 6 -c tessedit_char_whitelist=.0123456789")
    # string = image_to_string(otsu_thresh_image, lang="lets", config="--psm 7")
    float_value = re.findall("\d+\.\d+", string)  # this find the float values in the string
    if float_value:
        return float(float_value[0])
    else:
        return None


if __name__ == "__main__":
    directory = './image/RGB'
    filelist = get_files(directory)
    # print(filelist)

    for img_file in filelist:
        # print(img_file)
        img = cv.imread(img_file, 0)
        show_rgb(img)
        show_gray(img)
        clache(img)
        noise(img)
        out_img = remove_noise(img)
        img = img[530:630, 180:600]
        cv.imwrite('result/{}'.format(img_file), img)

    print('照片數量：', len(os.listdir(directory)), '張')

    mylist = [f for f in glob.glob("result/image/*.png")]

    for files in mylist:
        print(files)
        img = cv.imread(files)
        clean_display(img)
        process_image(img)
        value = ocr_image(img)
        print(value)

    print('照片數量：', len(mylist), '張')
