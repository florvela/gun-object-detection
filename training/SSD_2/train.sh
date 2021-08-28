python train.py \
configs/ssd300_vgg16_pascal-voc-07-12.json \
/content/gun-object-detection/training/SSD_2/dataset/output/images \
/content/gun-object-detection/training/SSD_2/dataset/output/labels \
--training_split=/content/gun-object-detection/training/SSD_2/dataset/output/train.txt \
--validation_split=/content/gun-object-detection/training/SSD_2/dataset/output/val.txt \
--label_maps=/content/gun-object-detection/training/SSD_2/dataset/output/label_maps.txt \
--learning_rate=0.001 \
--epochs=2 \
--batch_size=3 \
--shuffle=True \
--augment=True \
--output_dir=/content/drive/MyDrive/testing_pascal