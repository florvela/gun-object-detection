import os
import json
from tensorflow.keras.optimizers import SGD
from utils import ModelCheckpoint
from utils import training_utils

args = {
    'config': './configs/vgg16_flor.json',
    'images_dir': './datasets/',
    'labels_dir': './datasets/',
    'training_split': './datasets/train/split.txt',
    'validation_split':  './datasets/valid/split.txt',
    'label_maps': ['0', '1', 'Shotgun'],
    'checkpoint': None, #'/content/drive/MyDrive/ssd_train_output_32_batches/cp_ep_170_loss_5.7802.h5', #'/content/drive/MyDrive/ssd_train_output/cp_ep_100_loss_7.3720.h5', #'/content/drive/MyDrive/ssd_train_output/cp_ep_30_loss_11.7435.h5', #'./output_2/cp_ep_100_loss_17.3806.h5', # can be an existing h5 to load weights from and continue training
    'checkpoint_type': 'epoch',
    'checkpoint_frequency': 10,
    'initial_epoch': 0,
    'learning_rate': 0.001, #de aca https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    'epochs': 300,
    'batch_size': 64, # https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu
    'shuffle': True,
    'augment': False,
    'output_dir': '/mnt4/flor/git/otros/gun-object-detection/training/SSD/output/3_batches_001_lr_augmented'
}

# https://github.com/Socret360/object-detection-in-keras/tree/master/utils/ssd_utils

assert os.path.exists(args["config"]), "config file does not exist"
assert os.path.exists(args["images_dir"]), "images_dir does not exist"
assert os.path.exists(args["labels_dir"]), "labels_dir does not exist"
assert args["epochs"] > 0, "epochs must be larger than zero"
assert args["batch_size"] > 0, "batch_size must be larger than 0"
assert args["learning_rate"] > 0, "learning_rate must be larger than 0"
assert args["checkpoint_type"] in ["epoch", "iteration", "none"], "checkpoint_type must be one of epoch, iteration, none."

if not os.path.exists(args["output_dir"]):
    os.makedirs(args["output_dir"])

with open(args["config"], "r") as config_file:
    config = json.load(config_file)

training_data_generator, num_training_samples, validation_data_generator, num_validation_samples = training_utils.get_data_generator(config, args)
model = training_utils.get_model(config, args["label_maps"])
loss = training_utils.get_loss(config)
optimizer = SGD(
            learning_rate=args["learning_rate"],
            momentum=0.9,
            decay=0.0005,
            nesterov=False
        )


model.compile(optimizer=optimizer, loss=loss.compute)


num_iterations_per_epoch = num_training_samples//args["batch_size"]
if args["checkpoint_type"] == "epoch":
    assert args["checkpoint_frequency"] < args["epochs"], "checkpoint_frequency must be smaller than epochs."
elif args["checkpoint_type"] == "iteration":
    assert args["checkpoint_frequency"] < num_iterations_per_epoch * args["epochs"], "checkpoint_frequency must be smaller than num_iterations_per_epoch * args_epochs"

if args["checkpoint"] is not None:
    assert os.path.exists(args["checkpoint"]), "checkpoint does not exist"
    model.load_weights(args["checkpoint"], by_name=True)

model.fit(
    x=training_data_generator,
    validation_data=validation_data_generator,
    batch_size=args["batch_size"],
    validation_batch_size=args["batch_size"],
    epochs=args["epochs"],
    initial_epoch = args["initial_epoch"],
    callbacks=[
        ModelCheckpoint(
            initial_epoch = args["initial_epoch"],
            output_dir=args["output_dir"],
            epoch_frequency=args["checkpoint_frequency"] if args["checkpoint_type"] == "epoch" else None,
            iteration_frequency=args["checkpoint_frequency"] if args["checkpoint_type"] == "iteration" else None,
        )
    ]
)

