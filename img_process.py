import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import math


def image_plot(I, bool):
    plt.figure()
    plt.imshow(I, cmap='gray')
    plt.show(block= bool)

def image_save(I, f):

    os.chdir(r"/home/pnl/Downloads/numta/testing-e-modified")
    cv.imwrite(f, I)

def main( ):

    # root = r"/home/pnl/Downloads/numta/"
    # set_name = "training-e"
    # img_id = "e00225"
    # img_name = os.path.join(root, set_name, img_id + ".png")

    root= r"/home/pnl/Downloads/numta/testing-e"

    files= os.listdir(root)

    for f in files:

        img_gray=cv.imread(os.path.join(root,f),cv.IMREAD_GRAYSCALE)
        img_gray=255-img_gray
        img_res = cv.resize(img_gray, None, fx=0.75, fy=0.75, interpolation=cv.INTER_CUBIC)
        # (thresh, img_bin) = cv.threshold(img_gray, 128, 255, cv.THRESH_OTSU)
        #
        #
        # kernel = np.ones((5,5),np.uint8)
        #
        # img_dilate =cv.dilate(img_bin,kernel,iterations = 1)
        # # image_plot(img_dilate, False)
        #
        # img_erode=cv.erode(img_dilate,kernel,iterations = 1)
        # # image_plot(img_erode, False)


        # pad_x= [math.floor((180- img_erode.shape[0])/2), math.ceil((180- img_erode.shape[0])/2)]
        # pad_y= [math.floor((180- img_erode.shape[1])/2), math.ceil((180- img_erode.shape[1])/2)]


        img_mod= np.pad(img_res, ((100,100), (100,100)), 'constant', constant_values=255)
        img_res = cv.resize(img_mod, (180, 180), interpolation=cv.INTER_CUBIC)
        # image_plot(img_res, True)
        image_save(img_res, f)



if __name__ == "__main__":
    main()

