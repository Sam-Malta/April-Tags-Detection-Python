#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import os
from threading import Thread
import math

import numpy as np
from imutils.video import WebcamVideoStream
import cv2 as cv
from pupil_apriltags import Detector

# photoLocation = 'Examples'
# def get_examples():
#     examples = []
#     print("Loading Examples:")
#     for example in os.listdir(photoLocation):
#         examples.append(example)
#         print(example)
#     print("Done")
#     return examples

#listOfExampleNames = (get_examples())

#currentExample = listOfExampleNames[0]

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument("--families", type=str, default='tag16h5')
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--quad_decimate", type=float, default=1.0)
    parser.add_argument("--quad_sigma", type=float, default=0.8)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args

class WebcamVideoStream:
    def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
        args = get_args()
        self.stream = cv.VideoCapture(src)
        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        self.stream.set(cv.CAP_PROP_FPS, 10)
        (self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True




def solvePnP(imagePoints):  
    objectPoints = np.array([
        [0, 6.25, 0],
        [6.25, 6.25, 0],
        [6.25, 0.0, 0],
        [0, 0, 0],
    ], dtype=np.float32)

    cameraMatrix = np.array([[1165.0804701289262, 0.0, 627.0212940283434], [0.0, 1163.558769757253, 316.3709366006025], [0.0, 0.0, 1.0]])
    distCoeffs = np.array([[0.18586165886503364, -0.8772195344287632, -0.013994811668479672, -0.0077445107713434306, 0.3513116605396649]])
    retval, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags=cv.SOLVEPNP_IPPE_SQUARE)
    return rvec, tvec

def main():
    args = get_args()
    
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug
   
    cap = WebcamVideoStream(src=cap_device).start()
    # Detector
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 0
    start_time = time.time()
    
    while True:
        start_time = time.time()
        image = cap.read() # add ret and change to video cap if doesnt work
        # if not ret:
        #     break
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        tags = at_detector.detect(
            image,
            estimate_tag_pose = False,
            camera_params = [83.34434820, 148.1678191, 720//2, 1280//2], #CHANGES BASED ON CAM
            tag_size = 6.25,
            
        )

        debug_image = draw_tags(debug_image, tags, elapsed_time)
        elapsed_time = time.time() - start_time

        cv.imshow('AprilTag Detect Demo', debug_image)

        key = cv.waitKey(1)
        if key == 27: # ESC
            #currentExample = listOfExampleNames[listOfExampleNames.index(currentExample) + 1]
            break
    cap.stop()
    cv.destroyAllWindows()
            


def draw_tags(
    image,
    tags,
    elapsed_time,
):
    for tag in tags:
        if tag.hamming != 0 or tag.decision_margin <= 25: #Error thresholding
            continue
        
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        #Give Corners to PNP Solver
        rvex, tvex = solvePnP(corners)
        #print(f'X: {rvex[0]} Y: {rvex[1]} Z: {rvex[2]}')
        print(f'X: {tvex[0]} Y: {tvex[1]} Z: {tvex[2]}')

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

    #Highlighting Stuff
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()