import pretty_errors
import numpy as np
import cv2
import glob


def remove_blue(img):
    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 80, 0])
    upper_blue = np.array([160, 255, 255])
    # threshold
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_not(mask)

    # print(cv2.countNonZero(mask), mask.dtype)
    # Filter using contour area and remove small noise
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=len)
    # use the largest contour, in this case must be mango
    empty_img = mask.copy() * 0
    mask = cv2.drawContours(empty_img, [contours[-1]], -1, (255, 255, 255), -1)
    X, Y, W, H = cv2.boundingRect(contours[-1])
    X, Y, W, H = int(X / 1.1), int(Y / 1.1), int(W * 1.1), int(H * 1.1)
    # img[mask>0]=(0, 0, 0)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = img[Y:Y + H, X:X + W]

    # get the height and width
    width, height = get_dimensions(contours[-1])
    area = cv2.contourArea(contours[-1])
    return img, width, height, area


def remove_depthBG(img, nonzero_mean=False):
    # process the depth image
    # threshold
    # print(img.max(), img.mean())
    mask = img.copy()

    if nonzero_mean:
        nzmask = mask.flatten()
        # find non zero value
        nzmask = nzmask[np.nonzero(nzmask)]
        mask[mask < (nzmask.mean() - (nzmask.std() / 1.75))] = 0
    else:
        mask[mask < mask.mean()] = 0

    mask[mask < 0] = 255
    # mask = cv2.bitwise_not(mask)

    # print(cv2.countNonZero(mask), mask.dtype)
    # Filter using contour area and remove small noise
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=len)
    # use the largest contour, in this case must be mango
    empty_img = mask.copy() * 0
    mask = cv2.drawContours(empty_img, [contours[-1]], -1, (255, 255, 255), -1)

    X, Y, W, H = cv2.boundingRect(contours[-1])
    X, Y, W, H = int(X / 1.5), int(Y / 1.25), int(W * 1.15), int(H * 1.25)
    # img[mask>0]=(0, 0, 0)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = img[Y:Y + H, X:X + W]

    return img


def distance(pointA, pointB):
    # find Euclidean distance
    dist = np.linalg.norm(np.array(pointA) - np.array(pointB))
    return dist


def midpoint(ptA, ptB):
    # calculate midpoint using euclidean theorem
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def get_dimensions(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    # compute the Euclidean distance between the midpoints
    dA = distance((tltrX, tltrY), (blbrX, blbrY))  # distance 1
    dB = distance((tlblX, tlblY), (trbrX, trbrY))  # distance 2

    return sorted([dA, dB])  # sort by value smaller - to -larger so, width then height


def convert_to_mm(x, y, i):
    if i < 90:
        scale = 140 / (764 - 361)
    elif 89 < i < 120:
        scale = 140 / (779 - 372)
    elif 119 < i < 204:
        scale = 140 / (607 - 175)
    else:
        scale = 140 / (678 - 298)
    # print("scale : ", scale)
    return x * scale, y * scale


def get_files(directory, filetype='.png'):
    filelist = []
    for file_ in os.listdir(directory):
        if file_.endswith(filetype):
            filelist.append(os.path.join(directory, file_))

    return filelist


if __name__ == "__main__":
    # process the files
    color_files = glob.glob("dataset/100_100/rgb/*.png")
    depth_files = glob.glob("dataset/100_100/depth/*.png")

    for i in range(len(color_files)):

        h1, h2 = 20, 460
        if i < 120:
            w1, w2 = 260, 1030
        else:
            w1, w2 = 200, 970

        img = cv2.imread(color_files[i])
        depth = cv2.imread(depth_files[i], 0)
        depth = remove_depthBG(depth)
        # find the ratio between color and depth image resolutions
        hr, wr = img.shape[0] / depth.shape[0], img.shape[0] / depth.shape[0]
        # print(hr, wr)
        dh1, dh2, dw1, dw2 = int(h1 * 1.15 / hr), int(h2 / (hr * 1.15)), int(w1 * 1.15 / wr), int(w2 / (wr * 1.15))
        # print(dh1, dh2, dw1, dw2)
        img = img[h1:h2, w1:w2]
        depth = depth[dh1:dh2, dw1:dw2]
        # remove the blue background, corp the mango and get the height and width of mango
        img, width, height, area = remove_blue(img)

        # convert to mm values
        width, height = convert_to_mm(width, height, i)
        area, _ = convert_to_mm(area, area, i)
        dim = np.array([width, height])
        dim = np.around(dim, 2)  # round to two decimels
        print(f"{i}:, Width :{dim[0]}, Height: {dim[1]},  AREA: {area}")

        if i < 57:
            depth[depth < depth.max() - 25] = 0
        elif i in [61, 62, 63, 64, 65, 70, 71, 78, 79, ]:
            depth[depth < depth.max() - 15] = 0
        else:
            depth[depth < depth.max() - 18] = 0

        depth = remove_depthBG(depth, nonzero_mean=True)
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

        cv2.imwrite(f"dataset/100_100/processed/{i}.png", img)
        cv2.imwrite(f"dataset/100_100/processed/{i}d.png", depth)

        rgb = cv2.resize(img, (100, 100))
        d = cv2.resize(depth, (100, 100))
        buffer = np.zeros((100, 100, 4))
        buffer[:, :, :3] = rgb
        buffer[:, :, -1] = d
        np.savez(f"dataset/100_100/npz/{i}.npz", color=rgb, depth=d, rgbd=buffer, dimension=dim, areas=area)
        cv2.imshow('im', img)
        cv2.waitKey(1)
