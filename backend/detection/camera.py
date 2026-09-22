import cv2


class CameraCapture:
    """Simple wrapper around OpenCV webcam access with friendly error handling."""

    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Camera unavailable")
        return self.cap

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera unavailable")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Unable to read frame from camera")
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
