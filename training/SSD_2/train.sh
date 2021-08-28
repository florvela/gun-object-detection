python train.py \
configs/ssd300_vgg16_pascal-voc-07-12.json \
datasets/gunv3.v1i.voc/training_ready/images \
datasets/gunv3.v1i.voc/training_ready/labels \
--training_split=datasets/gunv3.v1i.voc/training_ready/train.txt \
--validation_split=datasets/gunv3.v1i.voc/training_ready/val.txt \
--label_maps=datasets/gunv3.v1i.voc/training_ready/label_maps.txt \
--learning_rate=0.001 \
--epochs=300 \
--batch_size=3 \
--shuffle=True \
--augment=True \
--output_dir=/content/drive/MyDrive/testing_pascal