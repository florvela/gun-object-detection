#!/bin/bash
python evaluate.py ./datasets/test/ predictions/complete_test_set/32_batches_0001_LR_not_augmented/ --output_dir evaluations/complete_test_set/32_batches_0001_LR_not_augmented --iou_threshold 0.4 &&
python evaluate.py ./datasets/test/ predictions/complete_test_set/32_batches_0001_LR_augmented/ --output_dir evaluations/complete_test_set/32_batches_0001_LR_augmented --iou_threshold 0.4 &&
python evaluate.py ./datasets/test/ predictions/complete_test_set/64_batches_0001_LR_not_augmented/ --output_dir evaluations/complete_test_set/64_batches_0001_LR_not_augmented --iou_threshold 0.4 &&
python evaluate.py ./datasets/test/ predictions/complete_test_set/64_batches_001_LR_not_augmented_100_epochs/ --output_dir evaluations/complete_test_set/64_batches_001_LR_not_augmented_100_epochs --iou_threshold 0.4 &&
python evaluate.py ./datasets/test/ predictions/complete_test_set/32_batches_001_lr_augmented/ --output_dir evaluations/complete_test_set/32_batches_001_LR_augmented --iou_threshold 0.4
