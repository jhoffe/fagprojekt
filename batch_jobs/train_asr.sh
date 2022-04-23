#!/bin/sh
cd /work3/s204071/fagprojekt
bsub < batch_jobs/train_asr_authentic.sh
bsub < batch_jobs/train_asr_synthetic.sh
bsub < batch_jobs/train_asr_mixed.sh
