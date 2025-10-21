# Machine Unlearning based on Trapdoors Equipped with a Distillator

This repository contains the implementation of our paper **"MUTED: Machine Unlearning based on Trapdoors Equipped with a Distillator"**, which proposes a novel trapdoor-based unlearning mechanism leveraging knowledge distillation.

## Repository Structure

```
.
├── flower/             # Training script of server and client models (federated setup)
├── distill/            # Key trigger fusing distillation
├── JS_compProb/        # Compute JS divergence, draw probability compare picture
├── dataset.txt         # Description of model architecture and datasets used
└── README.md           # Project documentation
```

## Getting Started

### Prerequisites

```
numpy==1.26.4  
torch>=2.0  
torchvision>=0.15  
flwr>=1.5  
matplotlib==3.10.7
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Training and Evaluation

To train the global model and perform unlearning experiments:

```bash
cd flower
python server.py
python client.py {i}
```

To perform trapdoor distillation:

- see distill/Readme.md

To visualize JS divergence comparison:

- see JS_compProb/Readme.md

## Experimental Details

* **Datasets:** MNIST, Fashion-MNIST, CIFAR-10
* **Model Architecture:** ResNet-18 implemented within the Flower federated learning framework

<!-- ## Citation

If you find this work useful, please cite:

```
@article{
}
``` -->