# OMA Loss System

The OMA loss system is designed to be:

- composable
- model-agnostic
- optimization-group aware
- easy to extend with custom losses

It supports both simple objectives like:

- `l1`
- `l2`
- `l1 + kl`

and more advanced objectives like:

- `l1 + lpips`
- `l1 + adv`
- `l1 + kl + lpips + adv`

---

## Core ideas

OMA losses are built from three layers:

1. **Loss terms**
   - Small atomic loss components such as `L1LossTerm`, `KLLossTerm`, `LPIPSLossTerm`.

2. **Loss composer**
   - Combines multiple terms and aggregates them by optimization group such as:
     - `main`
     - `disc`

3. **Recipes**
   - Convenience builders for common training objectives such as autoencoder losses.

---

## Shared training state

Losses operate on a shared `state` dictionary.

Example:

```python
state = {
    "input": x,
    "recon": x_rec,
    "posterior": posterior,
    "latent": z,
    "global_step": self.global_step,
    "split": "train",
}
```

Each loss term reads only the keys it needs.

This makes the loss API flexible across:

- autoencoders
- GANs
- diffusion models
- segmentation models
- custom research code

---

## Basic usage

### Compose losses manually

```python
from oma.losses import LossComposer
from oma.losses.terms import L1LossTerm, KLLossTerm

loss_fn = LossComposer([
    L1LossTerm(pred_key="recon", target_key="input", weight=1.0, name="l1"),
    KLLossTerm(posterior_key="posterior", weight=1e-6, name="kl"),
])
```

Then inside a method:

```python
state = {
    "input": x,
    "recon": x_rec,
    "posterior": posterior,
    "split": "train",
    "global_step": self.global_step,
}

out = loss_fn(state)
loss = out["losses"]["main"]
logs = out["logs"]
```

---

## Using a recipe

```python
from oma.losses.recipes import build_ae_l1_kl_loss

loss_fn = build_ae_l1_kl_loss(kl_weight=1e-6)
```

For a more advanced LDM-style autoencoder setup:

```python
from oma.losses.recipes import build_ldm_autoencoder_loss

loss_fn = build_ldm_autoencoder_loss(
    lpips_model=lpips_model,
    discriminator=discriminator,
    kl_weight=1e-6,
    perceptual_weight=1.0,
    adv_weight=0.5,
    disc_start=50001,
)
```

---

## Optimization groups

Each term belongs to an optimization group.

Common groups are:

- `main`
- `disc`

Example:

- reconstruction loss -> `main`
- KL loss -> `main`
- generator adversarial loss -> `main`
- discriminator adversarial loss -> `disc`

The composer aggregates them separately:

```python
out["losses"] == {
    "main": ...,
    "disc": ...,
}
```

This is what allows an OMA method to support either:

- automatic optimization
- manual optimization

depending on how many groups are active.

---

## Built-in terms

### Pixel losses
Located in `oma/losses/terms/pixel.py`

- `L1LossTerm`
- `L2LossTerm`
- `CharbonnierLossTerm`
- `HuberLossTerm`

### Regularization losses
Located in `oma/losses/terms/regularization.py`

- `KLLossTerm`
- `LatentL1LossTerm`
- `LatentL2LossTerm`
- `LogVarRegularizationTerm`

### Perceptual losses
Located in `oma/losses/terms/perceptual.py`

- `LPIPSLossTerm`
- `FeatureExtractorPerceptualLossTerm`

### Adversarial losses
Located in `oma/losses/terms/adversarial.py`

- `GeneratorAdversarialTerm`
- `DiscriminatorAdversarialTerm`
- `FeatureMatchingLossTerm`

---

## Writing your own loss term

To implement a custom loss, inherit from `LossTerm`.

Example:

```python
import torch
from oma.losses.base import LossTerm


class MyEdgeLoss(LossTerm):
    def __init__(self, pred_key="pred", target_key="target", weight=1.0, name="edge"):
        super().__init__(weight=weight, name=name, group="main")
        self.pred_key = pred_key
        self.target_key = target_key

    def validate(self, state):
        if self.pred_key not in state:
            raise KeyError(f"Missing key: {self.pred_key}")
        if self.target_key not in state:
            raise KeyError(f"Missing key: {self.target_key}")

    def compute(self, state):
        pred = state[self.pred_key]
        target = state[self.target_key]
        return torch.abs(pred - target).mean()
```

Then use it inside a composer:

```python
loss_fn = LossComposer([
    MyEdgeLoss(pred_key="recon", target_key="input", weight=0.2),
])
```

---

## Writing a stateful loss term

If your term needs to write intermediate values back into the shared state,
inherit from `StatefulLossTerm`.

This is useful for:

- adversarial logits
- cached features
- advanced diagnostics

---

## Grayscale medical images and LPIPS

LPIPS is usually designed for 3-channel natural images.

For grayscale medical images, `LPIPSLossTerm` supports:

- repeating 1 channel to 3 channels
- optional input normalization

Example:

```python
LPIPSLossTerm(
    lpips_model=lpips_model,
    pred_key="recon",
    target_key="input",
    repeat_grayscale=True,
)
```

For medical imaging, you may later prefer a domain-specific feature extractor
via `FeatureExtractorPerceptualLossTerm`.

---

## Design philosophy

OMA losses should be:

- explicit rather than magical
- composable rather than monolithic
- reusable across tasks
- easy to adapt for research

The loss system is designed so that users can start with simple recipes and
gradually move toward fully custom objectives without rewriting the training loop.