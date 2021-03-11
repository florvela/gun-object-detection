from networks import SSD_VGG16

def get_model(config, label_maps):
    model_config = config["model"]
    if model_config["name"] == "ssd_vgg16":
        return SSD_VGG16(
            config=config,
            label_maps=label_maps,
            is_training=True
        )
    else:
        print(f"model with name ${model_config['name']} has not been implemented yet")
        exit()
