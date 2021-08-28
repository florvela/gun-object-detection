python test.py \
datasets/training_ready/test.txt \
datasets/training_ready/images \
datasets/training_ready/labels \
configs/ssd300_vgg16_pascal-voc-07-12.json \
/content/drive/MyDrive/testing_pascal/cp_17_loss-5.91_valloss-5.78.h5 \
--label_maps=datasets/training_ready/label_maps.txt \
--output_dir=output/flor_v1 \
--num_predictions=3