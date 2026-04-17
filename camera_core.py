from picamera2 import Picamera2
import cv2
import time

class CameraManager:
    def __init__(self, exposure=1000, gain=3.0, fps=30):
        self.exposure = exposure
        self.gain = gain
        self.fps = fps
        self.picam2 = None

    def initialize_camera(self):
        self.picam2 = Picamera2()
        
        # We configure 'main' for high-res OCR, and 'lores' can be used for live preview if needed
        video_config = self.picam2.create_video_configuration(
            main={"size": (1456, 1088), "format": "BGR888"},
            lores={"size": (640, 480), "format": "BGR888"},
            controls={
                "FrameRate": self.fps,
                "ExposureTime": self.exposure,
                "AnalogueGain": self.gain
            }
        )
        self.picam2.configure(video_config)
        self.picam2.start()

    def close(self):
        if self.picam2:
            self.picam2.stop()

    def get_frame(self):
        """Captures a frame and returns it as a standard OpenCV BGR array."""
        if not self.picam2:
            return None
        try:
            # We use 'main' to get the high-resolution frame for better text extraction
            frame = self.picam2.capture_array("main")
            return frame
        except Exception as e:
            print("Error capturing frame:", e)
            return None
