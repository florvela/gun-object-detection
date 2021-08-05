
def get_config(model):
    if model == "yolov4":
        return {
            "model_type": "yolo",
            "input_width": 416,
            "input_height": 416,
            "confidence_threshold": 0.7,
            "nms_threshold": 0.5,
            "weights_path": "./yolo_utils/yolov4-custom-flor_best.weights",
            "cfg_path": "./yolo_utils/yolov4-custom-flor.cfg",
            "labels":{
                "KNIFE":0.7,
                "SHOTGUN":0.8,
                "RIFLE":0.8,
            }
        }
    elif model == "SSD_1":
        return {
            'config': './ssd_utils/configs/vgg16_flor.json',
            'weights_path': 'cp_ep_300_loss_7.1364.h5',
            'confidence_threshold': 0.8,
            'num_predictions': 10,
            'show_class_label': True,
            "labels": {
                "KNIFE": 0.7,
                "SHOTGUN": 0.8,
                "RIFLE": 0.8,
            }
        }

    return None