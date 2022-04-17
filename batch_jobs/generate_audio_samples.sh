#!/bin/sh
bsub < generate_audio_samples_test_clean.sh
bsub < generate_audio_samples_test_other.sh
bsub < generate_audio_samples_train_clean_100.sh
bsub < generate_audio_samples_train_clean_360.sh
bsub < generate_audio_samples_train_other_500.sh
