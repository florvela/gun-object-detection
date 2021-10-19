#!/bin/bash
python make_predictions.py ./output/32_batches_001_LR_augmented/cp_ep_100_loss_6.5782.h5 ../predictions/valid_test_set/ssd_32_0.001_sgd_100_yes/ ../datasets/valid/ &&

python make_predictions.py ./output/32_batches_0001_LR_augmented/cp_ep_100_loss_8.2067.h5 ../predictions/valid_test_set/ssd_32_0.0001_sgd_100_yes/ ../datasets/valid/ &&
python make_predictions.py ./output/32_batches_0001_LR_augmented/cp_ep_200_loss_7.3858.h5 ../predictions/valid_test_set/ssd_32_0.0001_sgd_200_yes/ ../datasets/valid/ &&
python make_predictions.py ./output/32_batches_0001_LR_augmented/cp_ep_300_loss_7.1364.h5 ../predictions/valid_test_set/ssd_32_0.0001_sgd_300_yes/ ../datasets/valid/ &&

python make_predictions.py ./output/32_batches_0001_LR_not_augmented/cp_ep_100_loss_6.2221.h5 ../predictions/valid_test_set/ssd_32_0.0001_sgd_100_no/ ../datasets/valid/ &&
python make_predictions.py ./output/32_batches_0001_LR_not_augmented/cp_ep_200_loss_5.5593.h5 ../predictions/valid_test_set/ssd_32_0.0001_sgd_200_no/ ../datasets/valid/ &&
python make_predictions.py ./output/32_batches_0001_LR_not_augmented/cp_ep_300_loss_5.3551.h5 ../predictions/valid_test_set/ssd_32_0.0001_sgd_300_no/ ../datasets/valid/ &&

python make_predictions.py ./output/64_batches_001_LR_not_augmented/cp_ep_100_loss_5.1822.h5 ../predictions/valid_test_set/ssd_64_0.001_sgd_100_no/ ../datasets/valid/ &&
python make_predictions.py ./output/64_batches_001_LR_not_augmented/cp_ep_200_loss_4.8415.h5 ../predictions/valid_test_set/ssd_64_0.001_sgd_200_no/ ../datasets/valid/ &&
python make_predictions.py ./output/64_batches_001_LR_not_augmented/cp_ep_300_loss_4.5963.h5 ../predictions/valid_test_set/ssd_64_0.001_sgd_300_no/ ../datasets/valid/ &&

python make_predictions.py ./output/64_batches_0001_LR_not_augmented/cp_ep_100_loss_13.9490.h5 ../predictions/valid_test_set/ssd_64_0.0001_sgd_100_no/ ../datasets/valid/ &&
python make_predictions.py ./output/64_batches_0001_LR_not_augmented/cp_ep_200_loss_6.1574.h5 ../predictions/valid_test_set/ssd_64_0.0001_sgd_200_no/ ../datasets/valid/ &&

python make_predictions.py ./output/32_batches_001_LR_augmented_adam/cp_ep_10_loss_4.9704.h5 ../predictions/valid_test_set/ssd_32_0.001_adam_10_yes/ ../datasets/valid/ &&
python make_predictions.py ./output/32_batches_001_LR_augmented_adam/cp_ep_20_loss_3.9508.h5 ../predictions/valid_test_set/ssd_32_0.001_adam_20_yes/ ../datasets/valid/ &&
python make_predictions.py ./output/32_batches_001_LR_augmented_adam/cp_ep_60_loss_3.1619.h5 ../predictions/valid_test_set/ssd_32_0.001_adam_60_yes/ ../datasets/valid/
