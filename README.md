# Misleading Clinical Information RLVR
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
## Data
Public Images and dataset can be found here: <ins link to code>




