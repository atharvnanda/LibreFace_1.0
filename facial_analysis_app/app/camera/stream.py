import cv2

class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
