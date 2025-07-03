# au_detector.py

import os
import random
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from libreface.AU_Recognition.solver_inference_combine import solver_inference_image_task_combine


class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class CustomSolver(solver_inference_image_task_combine):
    def run_pil(self, image_pil, task="au_recognition"):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        transformed_image = transform(image_pil)

        if task == "au_recognition":
            pred_labels = self.image_inference_au_recognition(transformed_image)
            pred_labels = pred_labels.squeeze().tolist()
            return dict(zip(self.au_recognition_aus, pred_labels))

        elif task == "au_detection":
            pred_labels = self.image_inference_au_detection(transformed_image)
            pred_labels = pred_labels.squeeze().tolist()
            return dict(zip(self.au_detection_aus, pred_labels))

        else:
            raise NotImplementedError(f"Unsupported task: {task}")


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def format_output(out_dict, task="au_recognition"):
    new_dict = {}
    for k, v in out_dict.items():
        if task == "au_recognition":
            new_dict[f"au_{k}_intensity"] = round(v, 3)
        elif task == "au_detection":
            new_dict[f"au_{k}"] = int(round(v))  # ensure 0 or 1
        else:
            raise NotImplementedError(f"format_output() not defined for the task - {task}")
    return new_dict


def init_solver_and_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    set_seed(opts.seed)
    solver = CustomSolver(opts).to(device)
    return solver, device


def get_au_data_from_frame(frame, solver):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    try:
        au_detected = solver.run_pil(pil_image, task="au_detection")
        au_intensity = solver.run_pil(pil_image, task="au_recognition")

        return format_output(au_detected, task="au_detection"), format_output(au_intensity, task="au_recognition")

    except Exception as e:
        print(f"[ERROR] AU Inference: {e}")
        return {}, {}
