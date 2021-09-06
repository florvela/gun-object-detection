import cv2
import os
import json
import pdb
import pickle
import numpy as np

def test(config):
    from utils import inference_utils

    weights, test_pred_filename = config
    print(weights,test_pred_filename)

    args = {
        'input_image': '../../src/test2.jpg',
        'config': './configs/vgg16_flor.json',
        'weights': weights,
        'test_pred_filename': test_pred_filename,
        'label_maps': ['KNIFE', 'SHOTGUN', 'RIFLE'],
        'thresholds': [0.8, 0.8, 0.8],
        'confidence_threshold': 0.5,
        'num_predictions': 10,
        'show_class_label': True
    }

    assert os.path.exists(args["input_image"]), "config file does not exist"
    assert os.path.exists(args["config"]), "config file does not exist"
    assert args["num_predictions"] > 0, "num_predictions must be larger than zero"

    with open(args["config"], "r") as config_file:
        config = json.load(config_file)

    input_size = config["model"]["input_size"]
    model_config = config["model"]

    model, process_input_fn = inference_utils.inference_ssd_vgg16(config, args)
    model.load_weights(args["weights"])

    label_maps = args["label_maps"]

    all_preds = []

    test_images_path = "./datasets/test/"
    predictions_path = "./datasets/predictions/789_sin_confidence/"
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)


    counter = 0
    for file in [file for file in os.listdir(test_images_path) if file.endswith(".jpg")]:
        counter += 1
        print("Testing image number",counter)

        image = cv2.imread(test_images_path+file)  # read image in bgr format
        image = np.array(image, dtype=np.float)
        image = np.uint8(image)
        image_height, image_width, _ = image.shape
        height_scale, width_scale = input_size/image_height, input_size/image_width
        image = cv2.resize(image, (input_size, input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = process_input_fn(image)
        image = np.expand_dims(image, axis=0)
        y_pred = model.predict(image)

        has_predictions = False
        for pred in y_pred[0]:
            # pdb.set_trace()
            class_ = pred[0].astype(float)
            confidence = pred[1].astype(float)
            if True: #confidence >= args["thresholds"][int(class_) - 1]:
                xmin = max(int(pred[2] / width_scale), 1)
                ymin = max(int(pred[3] / height_scale), 1)
                xmax = min(int(pred[4] / width_scale), image_width - 1)
                ymax = min(int(pred[5] / height_scale), image_height - 1)
                # line_elements = [class_, confidence, xmin, ymin, xmax, ymax]
                line_elements = [class_, confidence, xmin, ymin, xmax - xmin, ymax - ymin]
                # line_elements = [class_, confidence, int(pred[2]), int(pred[3]), int(pred[4]), int(pred[5])]
                line_elements = [str(e) for e in line_elements]
                line = " ".join(line_elements) + "\n"
                with open(predictions_path + file + '.txt', 'a') as write_file:
                    write_file.write(line)
                # pdb.set_trace()
                has_predictions = True
        if not has_predictions:
            with open(predictions_path + file + '.txt', 'a') as write_file:
                write_file.write("")

        # all_preds.append(y_pred[0])


    # test_predictions_filename = args["test_pred_filename"]
    # with open(test_predictions_filename, 'wb') as handle:
    #     pickle.dump(all_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)


test_configs = [
                ('./output/ssd_3_batches/cp_ep_80_loss_6.3691.h5','test_predictions_32_augmented.pickle'),
                # ('./output/ssd_train_output_32_batches_augmented/cp_ep_300_loss_7.1364.h5', 'test_predictions_32_augmented.pickle'),
                # ('./output/ssd_train_output_32_batches/cp_ep_300_loss_5.3551.h5', 'test_predictions_32.pickle'),
                # ('./output/ssd_train_output_32_batches/cp_ep_10_loss_14.1856.h5', 'test_predictions_32.pickle'),
                # ('./output/ssd_train_output_64_batches/cp_ep_280_loss_5.8012.h5', 'test_predictions_64.pickle')
                ]

for config in test_configs:
    test(config)



#
# for config in test_configs:
#     _, test_predictions_filename = config
#     all = pickle.load(open(test_predictions_filename, "rb"))
#     print(all)

