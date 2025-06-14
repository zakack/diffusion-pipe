# diffusion-pipe
A pipeline parallel training script for diffusion models.

Currently supports SDXL, Flux, LTX-Video, HunyuanVideo (t2v), Cosmos, Lumina Image 2.0, Wan2.1 (t2v and i2v), Chroma, HiDream, Stable Diffusion 3, Cosmos-Predict2.

**Work in progress.** This is a side project for me and my time is limited. I will try to add new models and features when I can.

## Features
- Pipeline parallelism, for training models larger than can fit on a single GPU
- Useful metrics logged to Tensorboard
- Compute metrics on a held-out eval set, for measuring generalization
- Training state checkpointing and resuming from checkpoint
- Efficient multi-process, multi-GPU pre-caching of latents and text embeddings
- Seemlessly supports both image and video models in a unified way
- Easily add new models by implementing a single subclass

## Recent changes
- 2025-06-14
  - Cosmos-Predict2 t2i LoRA training is supported. As usual, see the supported models doc for details.
  - Added option for using float8_e5m2 as the transformer_dtype.
- 2025-06-10
  - Stable Diffusion 3 LoRA training is supported.
  - Pinned Deepspeed version to fix error caused by Deepspeed 0.17.1.
- 2025-05-22
  - Add Automagic optimizer
  - Support i2v training for LTX-Video. Thanks @GallenShao for the PR!
  - Support multiple shuffling of tags when caching text embeddings. Credit to @gitmylo for the PR.
- 2025-05-07
  - Switch to official implementation of LTX-Video. Allows training the 13b LTX-Video model.
- 2025-04-19
  - Add support for first-frame-last-frame Wan model. Credit to @kabachuha for the PR.
  - Add wandb support. Credit to @ecarmen16 for the PR.
- 2025-04-18
  - Fix block swapping for HiDream. With ```blocks_to_swap = 24``` you can train rank 32 LoRA on a single 4090.
  - Support nf4 quantization for HiDream. With nf4 transformer, you can train LoRA on a single 4090 even without block swapping. See supported models doc for how to enable.
- 2025-04-15
  - Support HiDream.
- 2025-03-18
  - Add unsloth activation checkpointing. Reduces VRAM for a small performance hit.
  - Add partition_split option for manually controlling how layers are divided across multiple GPUs. Thanks @arczewski for the PR!
- 2025-03-16
  - Support loading any optimizer from the pytorch-optimizer library.
  - Wan transformer and UMT5 can now be loaded from ComfyUI files. Thanks to @qiwang1996 for the PR!

## Windows support
It will be difficult or impossible to make training work on native Windows. This is because Deepspeed only has [partial Windows support](https://github.com/microsoft/DeepSpeed/blob/master/blogs/windows/08-2024/README.md). Deepspeed is a hard requirement because the entire training script is built around Deepspeed pipeline parallelism. However, it will work on Windows Subsystem for Linux, specifically WSL 2. If you must use Windows I recommend trying WSL 2.

## Installing
Clone the repository:
```
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
```

If you alread cloned it and forgot to do --recurse-submodules:
```
git submodule init
git submodule update
```

Install Miniconda: https://docs.anaconda.com/miniconda/

Create the environment:
```
conda create -n diffusion-pipe python=3.12
conda activate diffusion-pipe
```

Install PyTorch first. As of this writing (May 5, 2025), you need PyTorch 2.6.0 CUDA 12.4 version (or earlier) for flash attention to work:
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install nvcc: https://anaconda.org/nvidia/cuda-nvcc. Probably try to make it match the CUDA version of PyTorch.

Install the rest of the dependencies:
```
pip install -r requirements.txt
```

### Cosmos requirements
NVIDIA Cosmos additionally requires TransformerEngine. This dependency isn't in the requirements file. Installing this was a bit tricky for me. On Ubuntu 24.04, I had to install GCC version 12 (13 is the default in the package manager), and make sure GCC 12 and CUDNN were set during installation like this:
```
CC=/usr/bin/gcc-12 CUDNN_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn pip install transformer_engine[pytorch]
```

## Dataset preparation
A dataset consists of one or more directories containing image or video files, and corresponding captions. You can mix images and videos in the same directory, but it's probably a good idea to separate them in case you need to specify certain settings on a per-directory basis. Caption files should be .txt files with the same base name as the corresponding media file, e.g. image1.png should have caption file image1.txt in the same directory. If a media file doesn't have a matching caption file, a warning is printed, but training will proceed with an empty caption.

For images, any image format that can be loaded by Pillow should work. For videos, any format that can be loaded by ImageIO should work. Note that this means **WebP videos are not supported**, because ImageIO can't load multi-frame WebPs.

## Supported models
See the [supported models doc](./docs/supported_models.md) for more information on how to configure each model, the options it supports, and the format of the saved LoRAs.

## Training
**Start by reading through the config files in the examples directory.** Almost everything is commented, explaining what each setting does. [This config file](./examples/main_example.toml) is the main example with all of the comments. [This dataset config file](./examples/dataset.toml) has the documentation for the dataset options.

Once you've familiarized yourself with the config file format, go ahead and make a copy and edit to your liking. At minimum, change all the paths to conform to your setup, including the paths in the dataset config file.

Launch training like this:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/hunyuan_video.toml
```
RTX 4000 series needs those 2 environment variables set. Other GPUs may not need them. You can try without them, Deepspeed will complain if it's wrong.

If you enabled checkpointing, you can resume training from the latest checkpoint by simply re-running the exact same command but with the `--resume_from_checkpoint` flag. You can also specify a specific checkpoint folder name after the flag to resume from that particular checkpoint (e.g. `--resume_from_checkpoint "20250212_07-06-40"`). This option is particularly useful if you have run multiple training sessions with different datasets and want to resume from a specific training folder.

Please note that resuming from checkpoint uses the **config file on the command line**, not the config file saved into the output directory. You are responsible for making sure that the config file you pass in matches what was previously used.

## Output files
A new directory will be created in ```output_dir``` for each training run. This contains the checkpoints, saved models, and Tensorboard metrics. Saved models/LoRAs will be in directories named like epoch1, epoch2, etc. Deepspeed checkpoints are in directories named like global_step1234. These checkpoints contain all training state, including weights, optimizer, and dataloader state, but can't be used directly for inference. The saved model directory will have the safetensors weights, PEFT adapter config JSON, as well as the diffusion-pipe config file for easier tracking of training run settings.

## Reducing VRAM requirements
The [wan_14b_min_vram.toml](./examples/wan_14b_min_vram.toml) example file has all of these settings enabled.
- Use AdamW8BitKahan optimizer:
  ```
  [optimizer]
  type = 'AdamW8bitKahan'
  lr = 5e-5
  betas = [0.9, 0.99]
  weight_decay = 0.01
  stabilize = false
  ```
- Use block swapping if the model supports it: ```blocks_to_swap = 32```
- Try the expandable_segments feature in the CUDA memory allocator:
  - ```PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /home/you/path/to/config.toml```
  - I've seen this help a lot when training on video with multiple aspect ratio buckets.
  - On my system, sometimes this causes random CUDA failures. If training gets through a few steps though, it will train indefinitely without failures. Very weird.
- Use unsloth activation checkpointing: ```activation_checkpointing = 'unsloth'```

## Parallelism
This code uses hybrid data- and pipeline-parallelism. Set the ```--num_gpus``` flag appropriately for your setup. Set ```pipeline_stages``` in the config file to control the degree of pipeline parallelism. Then the data parallelism degree will automatically be set to use all GPUs (number of GPUs must be divisible by pipeline_stages). For example, with 4 GPUs and pipeline_stages=2, you will run two instances of the model, each divided across two GPUs.

## Pre-caching
Latents and text embeddings are cached to disk before training happens. This way, the VAE and text encoders don't need to be kept loaded during training. The Huggingface Datasets library is used for all the caching. Cache files are reused between training runs if they exist. All cache files are written into a directory named "cache" inside each dataset directory.

This caching also means that training LoRAs for text encoders is not currently supported.

Two flags are relevant for caching. ```--cache_only``` does the caching flow, then exits without training anything. ```--regenerate_cache``` forces cache regeneration. If you edit the dataset in-place (like changing a caption), you need to force regenerate the cache (or delete the cache dir) for the changes to be picked up.

## Extra
You can check out my [qlora-pipe](https://github.com/tdrussell/qlora-pipe) project, which is basically the same thing as this but for LLMs.
