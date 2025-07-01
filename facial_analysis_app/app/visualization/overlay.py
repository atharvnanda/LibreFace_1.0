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


def draw_au_data(frame, aus: dict):
    y0, dy = 30, 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (au_name, value) in enumerate(aus.items()):
        y = y0 + i * dy
        label = f"{au_name}: {value:.2f}" if isinstance(value, float) else f"{au_name}: {value}"
        cv2.putText(frame, label, (10, y), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return frame