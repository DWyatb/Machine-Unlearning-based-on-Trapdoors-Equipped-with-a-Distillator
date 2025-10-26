# Key Trigger Fusing Distillation

This folder implements **Key Fusing Distillation**, a process to generate pseudo labels from old global model and train a new global model.

---

## Usage

### 1. `python gather_data(_key).py`
> Gather and prepare key fusing distillation data.

- Outputs `.npz` files containing all training x.  

### 2. `python distill_generate_y_pred.py`
> Extract label from old global model.

### 3. `python distill.py`
> Perform **Key Trigger Fusing Distillation**.
