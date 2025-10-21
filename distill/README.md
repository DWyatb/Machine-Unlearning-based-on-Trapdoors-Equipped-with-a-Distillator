# Key Trigger Fusing Distillation

This folder implements **Key Fusing Distillation**, a process to generate pseudo labels from old global model and train a new global model.

---

## Usage

### 1. `python gather_data.py`
> Gather and prepare key fusing distillation data.

- Outputs `.npz` files containing all training x.  

### 2. `python distill.py`
> Perform **Key Trigger Fusing Distillation**.

* models/resnet.py should be in the directery before xxx_JSdivergence.py is executed
