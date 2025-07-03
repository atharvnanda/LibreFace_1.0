# main_realtime.py

import threading
import cv2
from PIL import Image
import time

from facial_analysis_app.app.camera.stream import VideoStream
from facial_analysis_app.app.libreface_utils.au_detector import init_solver_and_device, get_au_data_from_frame
from facial_analysis_app.app.visualization.tkinter_ui import AUWindow


# --------------------------
# AU Inference + Webcam in Background Thread
# --------------------------
def run_realtime_inference(stream, solver, ui):
    print("[INFO] Background thread started.")
    while True:
        frame = stream.get_frame()
        if frame is None:
            continue

        # AU Analysis
        try:
            au_detected, au_intensity = get_au_data_from_frame(frame, solver)
            all_aus = {**au_detected, **au_intensity}
            ui.update_aus(all_aus)
        except Exception as e:
            print(f"[ERROR] AU Inference: {e}")

        # Show Webcam
        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stream.release()
            break


# --------------------------
# Main Setup
# --------------------------
if __name__ == "__main__":
    solver, device = init_solver_and_device()
    print(f"[INFO] Using device: {device}")
    stream = VideoStream()

    # Start Tkinter UI in main thread
    ui = AUWindow()

    # Run video+inference in background
    processing_thread = threading.Thread(target=run_realtime_inference, args=(stream, solver, ui))
    processing_thread.daemon = True
    processing_thread.start()

    print("[INFO] Launching UI...")
    ui.run()  # This blocks until UI is closed
    print("[INFO] Application exited.")
