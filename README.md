# Few-shot-RLVR-mismatch-clinical-description
This Github details reinforcement learning and transformers inference code for "The Impact of Misleading Clinical Information on Vision-Language Models and Human Experts in Medical Genetics"

## File Structure
```bash
├───data            
├───inference       
└───train
```
## Setup
### Train Environment
We recommend training with flash attention installed from a pre-built wheel. Find your prebuilt flash-attn wheel here: https://mjunya.com/flash-attention-prebuild-wheels/.  

Use the following commands to create your environment

```bash
conda create -y -n trl-flash python=3.13
conda activate trl-flash
pip install -r requirements-train.txt
wget <flash-attn-wheel>
cd <flash-attn-wheel>
pip install .
cd ..
```

### Inference Enviornment
If you do not have flash-attn installed, you may also run these commands:

```bash
conda create -y -n trl python=3.13
conda activate trl
pip install -r requirements-inference.txt
```
## Usage
### Training
To call training, make sure you have environment installed. The bash script provided allocates 2 a100 GPUs for 1 day. Please change the script accordingly to match with your existing set up. You will also need a huggingface token name saved on your system. 

Image root saved file path to the public images. EG "<image_root>/slide_images/slide_002.png"

To run training call: 
```bash
cd train
sbatch m27b_all_lin_grpo_reinforcement_CRO_1500.sh <hf_token_name> <image_root>
```
Inference from this directory is then: 
```bash
sbatch m27b_all_lin_grpo_inference_CRO_1500_mult_gpu.sh
```

### Inference 
To call inference, you can run: 
```bash 
cd inference 
sbatch m27b_all_lin_sft_grpo_inference_script_1500.sh
```

## Data
Public Images and dataset can be found [here](https://github.com/mariyakhan-0/VLM_mismatch_clinical_description) 






