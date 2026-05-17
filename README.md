# Code for Amazon Trainium Proposal for Optimizers
This repo contains the code for the proposal of Yifeng Liu adapted for Amazon Trainium chips, including MARS-M("mars_m_code/") and $\mu$ P for Gated Delta Network("gdn_mup_code/"). The code is adapted for Trainium chips with torch-neuronx and xla.

It is worth noting that you should use larger instances (like Trn1.32xlarge) with more memory to tun the code. Since the code is compatible with both Amazon Trainium chips and NVIDIA chips, you should avoid the command with "cuda".

To better install the torch-neuronx on your instances, first follow the instructions in [Quickstart](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/quick-start/index.html). Then check your torch version with:
```
python -c "import torch; import torch_neuronx; print(f'PyTorch: {torch.__version__}')"
```

If it's like `PyTorch: 2.9.0+cpu`, you are all set; otherwise (like `PyTorch: 2.9.0+cu:12.4`) you should uninstall torch by

```
pip uninstall -y torch
# (Optional: uninstall nvidia packages to save space)
pip list 2>/dev/null | grep -E '^nvidia-' | awk '{print $1}' | xargs -r pip uninstall -y
```

Then force to install CPU-version torch by
```
pip install torch==2.9.* --index-url https://download.pytorch.org/whl/cpu
# Then update Neuron packages:
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install --upgrade neuronx-cc==2.* torch-neuronx torchvision
```

Please be care that there would be a long time before training. And your program would be crack down if your use too large model that consumes much more memory than the chips can support.
