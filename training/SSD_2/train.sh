python train.py \
configs/ssd300_vgg16_pascal-voc-07-12.json \
datasets/training_ready/images \
datasets/training_ready/labels \
--training_split=datasets/training_ready/train.txt \
--validation_split=datasets/training_ready/val.txt \
--label_maps=datasets/training_ready/label_maps.txt \
--learning_rate=0.001 \
--epochs=300 \
--batch_size=32 \
--shuffle=True \
--augment=True \
--output_dir=/content/drive/MyDrive/testing_pascal