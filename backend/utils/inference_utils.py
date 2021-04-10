from networks import SSD_VGG16
from tensorflow.keras.applications import vgg16
from networks import SSD_MOBILENETV2
from tensorflow.keras.applications import mobilenet_v2


def inference_ssd_mobilenetv2(config, args):
    """"""
    model = SSD_MOBILENETV2(
        config,
        args["label_maps"],
        is_training=False,
        num_predictions=args["num_predictions"])
    process_input_fn = mobilenet_v2.preprocess_input

    return model, process_input_fn


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
        model, process_input_fn= inference_ssd_mobilenetv2(config, args)

    return model, process_input_fn