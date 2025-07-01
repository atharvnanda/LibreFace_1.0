import cv2
import torch
from PIL import Image
import numpy as np

from app.camera.stream import VideoStream
from app.visualization.overlay import draw_au_data
from app.libreface_utils.au_detector import CustomSolver, ConfigObject, set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Setup model configuration
opts = ConfigObject({
    'seed': 0,
    'ckpt_path': './facial_analysis_app/weights/au_recognition/combined_resnet.pt',
    'weights_download_id': "1CbnBr8OBt8Wb73sL1ENcrtrWAFWSSRv0",
    'image_inference': False,
    'au_recognition_data_root': '',
    'au_recognition_data': 'DISFA',
    'au_detection_data_root': '',
    'au_detection_data': 'BP4D',
    'fer_train_csv': 'training_filtered.csv',
    'fer_test_csv': 'validation_filtered.csv',
    'fer_data_root': '',
    'fer_data': 'AffectNet',
    'fold': 'all',
    'image_size': 256,
    'crop_size': 224,
    'au_recognition_num_labels': 12,
    'au_detection_num_labels': 12,
    'fer_num_labels': 8,
    'sigma': 10.0,
    'jitter': False,
    'copy_classifier': False,
    'model_name': 'resnet',
    'dropout': 0.1,
    'ffhq_pretrain': '',
    'hidden_dim': 128,
    'fm_distillation': False,
    'num_epochs': 30,
    'interval': 500,
    'threshold': 0,
    'batch_size': 256,
    'learning_rate': 3e-5,
    'weight_decay': 1e-4,
    'clip': 1.0,
    'when': 10,
    'patience': 5,
    'device': device
})

# Initialize solver
set_seed(opts.seed)
solver = CustomSolver(opts).to(device)

# Open webcam stream
stream = VideoStream()
print("Starting AU detection...")

while True:
    frame = stream.get_frame()
    if frame is None:
        print("No frame received. Exiting...")
        break

    try:
        # Convert frame to PIL
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Use custom solver logic (with transform)
        au_detected = solver.run_pil(pil_image, task="au_detection")
        au_intensity = solver.run_pil(pil_image, task="au_recognition")

        # Merge both
        all_aus = {**{
            f"au_{k}": v for k, v in au_detected.items()
        }, **{
            f"au_{k}_intensity": round(v, 3) for k, v in au_intensity.items()
        }}

        # Annotate and display
        annotated = draw_au_data(frame.copy(), all_aus)
        cv2.imshow("AU Detection (Realtime)", annotated)

    except Exception as e:
        print(f"Error during inference: {e}")
        cv2.imshow("AU Detection (Realtime)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stream.release()
print("Stopped.")
