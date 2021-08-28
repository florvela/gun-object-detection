python evaluate.py \
datasets/training_ready/images \
datasets/training_ready/labels \
output/cp_229_loss-5.06_valloss-5.17.h5 \
datasets/training_ready/test.txt \
--label_maps=datasets/training_ready/label_maps.txt \
--output_dir=output/evaluations/cp_229_loss-5.06_valloss-5.17.h5 \
--iou_threshold=0.5