import numpy as np
import cv2
from time import sleep, time
from picamera import PiCamera
import sys
import os
import random
import argparse
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default ="", help="Path to Image File")
parser.add_argument("--k_s_closing", "-c", type=int, default=110, help="Dimension of Square Kernel Used for Closing")
parser.add_argument("--threshold", "-t", type=int, default=150, help="Threshold to Create Binary Mask")
parser.add_argument("--k_s_median", "-m", type=int, default=75, help="Dimension of Square Kernel for Median Filter")
parser.add_argument("--capture", type=int, default=0, choices=[0, 1], help="1 to Capture Live")

# values of -t, -m, -c can be changed to suit different setups and lighting
# value of -t is most important

args = parser.parse_args()

def main(args):
    filename = []
    img = []
    
    # --capture 1 for realtime system
    if args.capture == 1:
        camera = PiCamera()
        camera.resolution = (3000, 3000) # gpu memory dist increase required for res > 2000
        camera.start_preview()
        #sleep(5)
        input("Press enter to continue...")
        ct = datetime.datetime.now()
        filename = f'{ct.year}_{ct.month}_{ct.day}_{ct.hour}_{ct.minute}_{ct.second}.jpg'
        camera.capture(f'imgs/{filename}') # saves every image. change before deployment
        camera.stop_preview()
        camera.close()
        img = cv2.imread(f'imgs/{filename}')

    else:
        # use image saved in img_path
        img = cv2.imread(args.img_path)
        if img is None:
            sys.exit("Could not read the image.")

    img = img[500:2500, 500:2500] # crops area around ROI. use smaller img to make script faster
    start_time = time()

    # visualize img after cropping
    #imS = cv2.resize(img, (840, 840)) 
    #winname = "Image"
    #cv2.namedWindow(winname)
    #cv2.moveWindow(winname, 100,100)
    #cv2.imshow(winname, imS)
    #cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Taking a matrix of size args.k_s_closing as the kernel
    kernel = np.ones((args.k_s_closing, args.k_s_closing), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) # to remove small dark spots I think

    #imS = cv2.resize(closing, (1080, 840)) 
    #winname = "Closing"
    #cv2.namedWindow(winname)
    #cv2.moveWindow(winname, 100,30)
    #cv2.imshow(winname, imS)
    #cv2.waitKey(0)

    ret, thresh = cv2.threshold(closing, args.threshold, 255, 1)
    median = cv2.medianBlur(thresh, args.k_s_median) # to blur texture while maintaining boundaries and edges

    contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #imS = cv2.resize(thresh, (1080, 840)) 
    #winname = "Binary"
    #cv2.namedWindow(winname)
    #cv2.moveWindow(winname, 100,30)
    #cv2.imshow(winname, imS)
    #cv2.waitKey(0)

    #imS = cv2.resize(median, (1080, 840)) 
    #winname = "Median"
    #cv2.namedWindow(winname)
    #cv2.moveWindow(winname, 100,30)
    #cv2.imshow(winname, imS)
    #cv2.waitKey(0)

    # use area to filter out too big or too small contours
    area = [cv2.contourArea(cnt) for cnt in contours]
    #print(area)
    cnt = contours[np.argmax([a if a > 500_000 and a < 1_600_000 else 0 for a in area])] # range depends on the distannce between the camera and the adapter. Reduce this range before deployment

    # do not need convex hull. delete before deployment
    hull = cv2.convexHull(cnt)
    area_hull = cv2.contourArea(hull)
    cv2.drawContours(img, [hull], 0, (0, 0, 255), 2) # plot the hull

    area_cnt = area[np.argmax([a if a > 500_000 and a < 1_600_000 else 0 for a in area])]
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2) # plot the contour

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(img, center, radius, (255, 0, 0), 2) # plot the min enclosing circle

    #print(f'Area: {area[-10:]}')
    #print(f'File: {args.img_path}    Diameter_Enc: {radius*2}	Diameter_Cnt: {np.sqrt(area_cnt/(np.pi))*2}    Diameter_hull: {np.sqrt(area_hull/np.pi)*2}')
    print(f'File: {args.img_path}    Diamter_Enc: {radius*2}px or {radius*2*0.0018587}')
    print(f'Time taken: {time() - start_time}')

    imS = cv2.resize(img, (1080, 840)) 
    winname = "Detected Circle"
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 800,30)
    cv2.imshow(winname, imS)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    return radius*2, area_cnt


if __name__ == "__main__":
    #dia_arr = []
    #area_cnt_arr = []
    count = 0
    while True:
        dia, area = main(args)
        #dia_arr.append(dia)
        #area_cnt_arr.append(area)
        #count += 1
        #print(f'Counter: {count}    Diameter_enc: {sum(dia_arr)/count}, Diameter_cnt: {np.sqrt((sum(area_cnt_arr)/count)/np.pi)}')
        #sleep(3600)
