#!/usr/bin/env python3

"""Acquire images from CSI camera

Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
NVIDIA Jetson Nano Developer Kit using OpenCV drivers for the camera 
and OpenCV are included in the base image

gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Defaults to 3280x2464 @ 15fps
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of the window on the screen

Reference: https://github.com/JetsonHacksNano/CSI-Camera

TODO(Kyungwon): Display two images in a separate thread
"""

import cv2
import os
import threading
import numpy as np


class CSICamera:

    def __init__ (self) :
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string: str):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running=True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running=False
        self.read_thread.join()

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed=grabbed
                    self.frame=frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened
        

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed=self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

# Currently there are setting frame rate on CSI Camera on Xavier NX through 
# gstreamer. Here we directly select sensor_mode 3 (1280x720, 59.9999 fps)
def gstreamer_pipeline(sensor_id: int = 0,
                       sensor_mode: int = 3,
                       capture_width: int = 3280,
                       capture_height: int = 2464,
                       display_width: int = 1280,
                       display_height: int = 720,
                       framerate: int = 1,
                       flip_method: int = 0):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        ))

def start_cameras():
    left_camera = None
    right_camera = None

    left_camera = CSICamera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            sensor_mode=3,
            flip_method=0,
            display_height=540,
            display_width=960,
        )
    )
    left_camera.start()

    right_camera = CSICamera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            sensor_mode=3,
            flip_method=0,
            display_height=540,
            display_width=960,
        )
    )
    right_camera.start()

    cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)

    if (
        not left_camera.video_capture.isOpened()
        or not right_camera.video_capture.isOpened()
    ):
        # Cameras did not open, or no camera attached

        print("Unable to open any cameras")
        # TODO: Proper cleanup
        SystemExit(0)

    while cv2.getWindowProperty("CSI Cameras", 0) >= 0 :
        
        _ , left_image=left_camera.read()
        _ , right_image=right_camera.read()
        camera_images = np.hstack((left_image, right_image))
        cv2.imshow("CSI Cameras", camera_images)

        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    machine = os.uname().machine

    if machine == "aarch64":
        print("This system is identified as one of the NVIDIA embedded platform.")
        start_cameras()
    elif machine == "armv7l":
        print("This system looks like Raspberry Pi.")
        pass
    else:
        print("This system can not be identified.")
        pass
    
