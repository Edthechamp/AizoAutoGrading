#this is opencv logic and answer extraction
import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

class CameraThread(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.paused = False
        self.latest_frame = None  # cache the latest raw frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.latest_frame = frame  # always update, regardless of pause

            if not self.paused:
                self.frame_ready.emit(frame)
            time.sleep(1/30)

        cap.release()
    
    def toggle_pause(self):
        self.paused = not self.paused