# MARS-M: When Variance Reduction Meets Matrices

This repository contains the official code for MARS-M paper.

## MARS-M

**MARS-M** is a brand-new optimizer that integrates matrix-based optimizer (i.e., Muon and Moonlight) with the variance-reduction based optimizer MARS to reduce high stochastic gradient variance in the training process.

We propose **MARS-M** that applies MARS to matrix-based optimizers (See  `optimizers/mars_m.py` for the implementation):

---

**Algorithm 1** MARS-M

---

$$
\begin{align*}
&\pmb{input: }\mathbf{X}_0\in\mathbb{R}^{A\times B}, \lambda, \beta, \{\gamma_t\}, \{\eta_t\}\\
&\text{Set }\mathbf{M}_0\leftarrow \mathbf{0}\textbf{ and }\mathbf{X}_1\leftarrow\mathbf{X}_0\\
&\pmb{for }\textbf{ }t=1,\pmb{ to }\textbf{ }n\textbf{ }\pmb{ do}\\
&\qquad\textbf{sample }\mathbf{\xi}_t\textbf{ and let }\mathbf{C}_t = \nabla f(\mathbf{X}_t, \mathbf{\xi}_t)+\gamma_t\bigg(\frac{\beta}{1-\beta}\bigg)\big(\nabla f(\mathbf{X}_t, \mathbf{\xi}_t)-\nabla f(\mathbf{X}_{t-1}, \mathbf{\xi}_t)\big)\\
&\qquad\mathbf{M}_t = \beta \mathbf{M}_{t-1} + (1-\beta)\text{Clip}(\mathbf{C}_t, 1)\\
&\qquad\mathbf{O}_t = \text{NewtonSchulz}(\mathbf{M}_t)\\
&\qquad\mathbf{X}_{t+1} = \mathbf{X}_t - \eta_t(0.2\cdot\mathbf{O}_t\cdot\sqrt{\max(A,B)} +  \lambda \mathbf{X}_t)\\
&\pmb{end}\textbf{ }\pmb{for}
\end{align*}
$$

---

To accelerate training process, we also propose the approximated version of MARS-M by substituting $f(\mathbf{X}\_{t-1}, \mathbf{\xi}\_t)$ with $f(\mathbf{X}\_{t-1}, \mathbf{\xi}\_{t-1})$ as follows:

---

**Algorithm 2** MARS-M-approx

---

$$
\begin{align*}
&\pmb{input: }\mathbf{X}_0\in\mathbb{R}^{A\times B}, \lambda, \beta, \{\gamma_t\}, \{\eta_t\}\\
&\text{Set }\mathbf{M}_0\leftarrow \mathbf{0}\textbf{ and }\mathbf{X}_1\leftarrow\mathbf{X}_0\\
&\pmb{for }\textbf{ }t=1,\pmb{ to }\textbf{ }n\textbf{ }\pmb{ do}\\
&\qquad\textbf{sample }\mathbf{\xi}_t\textbf{ and let }\mathbf{C}_t = \nabla f(\mathbf{X}_t, \mathbf{\xi}_t)+\gamma_t\bigg(\frac{\beta}{1-\beta}\bigg)\big(\nabla f(\mathbf{X}_t, \mathbf{\xi}_t)-\nabla f(\mathbf{X}_{t-1}, \mathbf{\xi}_{t-1})\big)\\
&\qquad\mathbf{M}_t = \beta \mathbf{M}_{t-1} + (1-\beta)\text{Clip}(\mathbf{C}_t, 1)\\
&\qquad\mathbf{O}_t = \text{NewtonSchulz}(\mathbf{M}_t)\\
&\qquad\mathbf{X}_{t+1} = \mathbf{X}_t - \eta_t(0.2\cdot\mathbf{O}_t\cdot\sqrt{\max(A,B)} +  \lambda \mathbf{X}_t)\\
&\pmb{end}\textbf{ }\pmb{for}
\end{align*}
$$

---



## Training GPT-2 from Scratch:

### Install Dependencies

```
$ pip install torch==2.1.2 transformers==4.33.0 datasets tiktoken numpy==1.26.4 wandb
```

### Data Preparation

```
$ python data/openwebtext/prepare.py
```

### **Start Training**

To train a model using the **MARS-M** optimizer, run the following command:

```bash
$ torchrun --standalone --nproc_per_node=8 train_mars_m.py config/${your_config_file}
```

This command initiates the training of a GPT-2 model on the OpenWebText dataset using the **MARS-M** optimizer. All relevant hyperparameters—training, model, and optimizer—are specified in the configuration file (`${your_config_file}`). These parameters can be adjusted directly in the configuration file or through the bash script.

### **Hyperparameter Details**

#### **Model Hyperparameters**:

- **n_layer**: Layers of networks, 12 for GPT2 Small, 24 for GPT2 Medium, 36 for GPT2 Large
- **n_head**: Number of heads, 12 for GPT2 small, 16 for GPT2 Medium, 20 for GPT2 Large
- **n_embd**: Embedding dimension, 768 for GPT2 small, 1024 for GPT2 Medium, 1280 for GPT2 Large

#### **Optimizer Hyperparameters**:

- **`learning_rate`**: Learning rate for the **MARS-M** optimizer.
- **`weight_decay`**: Weight decay for the **MARS-M** optimizer.
- **`beta1`**: momentum for **MARS-M** optimizer.

  - Default: `beta1=0.95, beta2=0.99`
- **`betas_1d`**: Weights for exponential moving average in AdamW optimizer (for 1d parameters).

  - Default: `(0.9, 0.95)`
- **`is_approx`**: Whether to use approximate gradient calculation (**MARS-M**-approx).

  - Default: `True`
- **`gamma`**: The scaling parameter that controls the strength of gradient correction.

  - Default: 0.025

#### **Training Hyperparameters**:

- **`batch_size`**: Mini-batch size per device. (for example GPT-2 Small on an A100 GPU typically uses a batch size of 15.)
- **`gradient_accumulation_steps`**: Gradient accumulation steps to ensure the total effective batch size matches the desired scale. (for example, for a total batch size of 480: $15 \times 4 \times 8 \, \text{GPUs}$.)
- **`schedule`**: learning rate schedule.
  - Default: `cosine`

For more detailed hyperparameter examples, refer to:

- `config/train_gpt2_small_mars_m.py`
- `scripts/run_mars_m_small.sh`

---

### Reproducing Our Results

#### **Reproducing GPT-2 Small (125M) Results**

Training with MARS-M using

```
$ bash scripts/run_mars_m_small.sh
```

or

```
$ torchrun --standalone --nproc_per_node=8 \
      train_mars_m.py \
      config/train_gpt2_small_mars_m.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4
```

#### Reproducing GPT2 Medium (355M) Results

Training with MARS-M using

```
$ bash scripts/run_mars_m_medium.sh
```

or

```
$ torchrun --standalone --nproc_per_node=8 \
      train_mars_m.py \
      config/train_gpt2_medium_mars_m.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4
```

#### Reproducing GPT2 Large (770M) Results

Training with MARS-M using

```
$ bash scripts/run_mars_m_large.sh
```

or

```
$ torchrun --standalone --nproc_per_node=8 \
      train_mars_m.py \
      config/train_gpt2_large_mars_m.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12
```

#### **Reproducing GPT-2 XL (1.5B) Results on FineWeb-Edu**

```
$ bash scripts/run_mars_m_xl_fw.sh
```

or

```
$ torchrun --standalone --nproc_per_node=8 \
      train_mars_m_fw.py \
      config/train_gpt2_xl_mars_m.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12
```

#### Reproducing Baseline Results

To reproduce the Moonlight baseline:

```
bash scripts/run_moonlight_{small/medium/large}.sh
```

Other baselines can be implemented with codes in `../MARS` folder.

Please adjust ``nproc_per_node``, ``batch_size``, and ``gradient_accumulation_steps`` accordingly if you use other hardware setup. Make sure their product equals 480.
