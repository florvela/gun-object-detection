from networks import SSD_VGG16
from tensorflow.keras.applications import vgg16

def inference_ssd_vgg16(config, args):
    model = SSD_VGG16(
        config,
        args["label_maps"],
        is_training=False,
        num_predictions=args["num_predictions"])
    process_input_fn = vgg16.preprocess_input

    return model, process_input_fn


def get_inference_model(config, args):
    model_config = config["model"]
    if model_config["name"] == "ssd_vgg16":
        model, process_input_fn = inference_ssd_vgg16(config, args)
    else:
        return None, None
    return model, process_input_fn