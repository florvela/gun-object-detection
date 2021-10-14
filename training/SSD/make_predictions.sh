#!/bin/bash
python make_predictions.py ./output/32_batches_001_LR_augmented/cp_ep_100_loss_6.5782.h5 ../predictions/complete_test_set/32_batches_001_LR_augmented_100_epochs/ ./datasets/test/ &&

python make_predictions.py ./output/32_batches_0001_LR_augmented/cp_ep_100_loss_8.2067.h5 ../predictions/complete_test_set/32_batches_0001_LR_augmented_100_epochs/ ./datasets/test/ &&
python make_predictions.py ./output/32_batches_0001_LR_augmented/cp_ep_200_loss_7.3858.h5 ../predictions/complete_test_set/32_batches_0001_LR_augmented_200_epochs/ ./datasets/test/ &&
python make_predictions.py ./output/32_batches_0001_LR_augmented/cp_ep_300_loss_7.1364.h5 ../predictions/complete_test_set/32_batches_0001_LR_augmented_300_epochs/ ./datasets/test/ &&

python make_predictions.py ./output/32_batches_0001_LR_not_augmented/cp_ep_100_loss_6.2221.h5 ../predictions/complete_test_set/32_batches_0001_LR_not_augmented_100_epochs/ ./datasets/test/ &&
python make_predictions.py ./output/32_batches_0001_LR_not_augmented/cp_ep_200_loss_5.5593.h5 ../predictions/complete_test_set/32_batches_0001_LR_not_augmented_200_epochs/ ./datasets/test/ &&
python make_predictions.py ./output/32_batches_0001_LR_not_augmented/cp_ep_300_loss_5.3551.h5 ../predictions/complete_test_set/32_batches_0001_LR_not_augmented_300_epochs/ ./datasets/test/ &&

python make_predictions.py ./output/64_batches_001_LR_not_augmented/cp_ep_100_loss_5.1822.h5 ../predictions/complete_test_set/64_batches_001_LR_not_augmented_100_epochs/ ./datasets/test/ &&
python make_predictions.py ./output/64_batches_001_LR_not_augmented/cp_ep_200_loss_4.8415.h5 ../predictions/complete_test_set/64_batches_001_LR_not_augmented_200_epochs/ ./datasets/test/ &&
python make_predictions.py ./output/64_batches_001_LR_not_augmented/cp_ep_300_loss_4.5963.h5 ../predictions/complete_test_set/64_batches_001_LR_not_augmented_300_epochs/ ./datasets/test/ &&

python make_predictions.py ./output/64_batches_0001_LR_not_augmented/cp_ep_100_loss_13.9490.h5 ../predictions/complete_test_set/64_batches_0001_LR_not_augmented_100_epochs/ ./datasets/test/ &&
python make_predictions.py ./output/64_batches_0001_LR_not_augmented/cp_ep_200_loss_6.1574.h5 ../predictions/complete_test_set/64_batches_0001_LR_not_augmented_200_epochs/ ./datasets/test/
