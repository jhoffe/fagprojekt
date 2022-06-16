# 02466 - Project work: Generating Speech from Transcripts
This repository contains the all the code for reproducing the results of the paper "Generating Speech from Transcripts".
Overall, the repository contains three main parts:
- `synthetic_gen/`: The TTS pipeline, which uses Tacotron2 and Waveglow to generate speech.
- `asr_model/`: The ASR model, which produces our evaluation results for our synthetic speech.
- `WaveNIST/`: Contains the code for creating an ARM, which can generate images of MNIST digits.

Each of these directories also contains another README, which will explain the files in the directory more in depth.

Moreover, the repo contains a few more directories, which solve some smaller tasks:
- `batch_jobs/`: The batch jobs folder contains all the shell scripts used to deploy the generation, training and evaluation to DTU's HPC environment.
- `scripts/`: This directory contains a few scripts to generate the datalists required for the ASR model to load the data.
- `upsampler/`: The upsampler directory contains a single script, which is used for upsampling the speech in LibriSpeech from 16000Hz to 22050Hz.

## Requirements for running code
Most of this code requires quite a lot of compute power to actually effectively train and forward pass the models.
For this paper most of the code was run on DTU's HPC environment with either NVIDIA A100 or V100 GPUs.
Moreover, some of the tasks are sequential, which means you can't actually run a part of the
project before you have run the previous. But at the very least, you should have:
- An at least 4-core processor
- A CUDA compatible GPU.

## How to get started
Since, it is basically impossible to actually run this code on a personal computer, only the pipeline for
dispatching the jobs to HPC will be described:

### TTS and ASR
First, you will have to clone repository:
```shell
git clone https://github.com/jhoffe/fagprojekt && cd fagprojekt
```
Afterwards, you should create a virtual environment with the script `batch_jobs/create_venv.sh`:
```shell
./batch_jobs/create_venv.sh
```
This script will create a virtual python environment in the repo folder, with all python
dependencies, including PyTorch with CUDA.

Now you can begin dispatching the jobs. First, you should generate the synthetic audio samples:
```shell
./batch_jobs/generate_audio_samples.sh
```
When all of these have finished then you can begin upsampling all the LibriSpeech data
(it will automatically download it as well):
```shell
bsub < batch_jobs/upsample_training_audio.sh
```
Then create the datalists:
```shell
bsub < batch_jobs/create_datalists.sh
```
Afterwards, you should generate the spectrograms for each audio file:
```shell
./batch_jobs/spec_gen.sh
```
Finally, you should be able to actually train the models:
```shell
./batch_jobs/train_asr.sh
```
Moreover, you can try finetuning the authentic model with the mixed dataset by running the following command:
```shell
bsub < batch_jobs/finetune_asr_authentic.sh
```

If you get errors with `wandb`, then you should either create an account and an API key 
and replace it, or you can add the command to disable it completely:
```shell
export WANDB_MODE=disabled
```
or if you want to log it locally:
```shell
export WANDB_MODE=offline
```

### WaveNIST
The WaveNIST part of the project is separate from the rest, so it can actually be run
independently. It consists of two models, the first one with a discretized version and
a second one with continuous input and output.

For the discretized the command is:
```shell
bsub < batch_jobs/train_wavenist_dgm.sh
```
And for the continuous version:
```shell
bsub < batch_jobs/train_wavenist_v5.sh
```