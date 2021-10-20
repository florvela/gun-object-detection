#!/bin/bash
python evaluate.py ../predictions/complete_test_set/ ../evaluations/complete_test_set/ &&
python evaluate.py ../predictions/valid_test_set/ ../evaluations/valid_test_set/ --set valid