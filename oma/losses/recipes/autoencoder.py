from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..composer import LossComposer
from ..terms import (
    L1LossTerm,
    L2LossTerm,
    CharbonnierLossTerm,
    HuberLossTerm,
    KLLossTerm,
    LPIPSLossTerm,
    GeneratorAdversarialTerm,
    DiscriminatorAdversarialTerm,
    FeatureMatchingLossTerm,
)


def _build_recon_term(
    recon_loss: str,
    pred_key: str,
    target_key: str,
    weight: float,
):
    recon_loss = recon_loss.lower()

    if recon_loss == "l1":
        return L1LossTerm(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name="l1",
        )
    if recon_loss == "l2":
        return L2LossTerm(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name="l2",
        )
    if recon_loss == "charbonnier":
        return CharbonnierLossTerm(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name="charbonnier",
        )
    if recon_loss == "huber":
        return HuberLossTerm(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name="huber",
        )

    raise ValueError(
        f"Unsupported recon_loss='{recon_loss}'. "
        "Choose from ['l1', 'l2', 'charbonnier', 'huber']."
    )


def build_autoencoder_loss(
    recon_loss: str = "l1",
    recon_weight: float = 1.0,
    use_kl: bool = True,
    kl_weight: float = 1e-6,
    use_lpips: bool = False,
    lpips_model: Optional[nn.Module] = None,
    lpips_weight: float = 1.0,
    repeat_grayscale_for_lpips: bool = True,
    normalize_lpips_inputs: bool = False,
    use_adversarial: bool = False,
    discriminator: Optional[nn.Module] = None,
    adv_weight: float = 1.0,
    disc_weight: float = 1.0,
    disc_start: int = 0,
    disc_mode: str = "hinge",
    use_feature_matching: bool = False,
    feature_matching_weight: float = 0.0,
    conditional: bool = False,
    cond_key: Optional[str] = None,
    pred_key: str = "recon",
    target_key: str = "input",
    posterior_key: str = "posterior",
) -> LossComposer:
    """
    General autoencoder loss builder for OMA.

    Supports:
        - reconstruction only
        - reconstruction + KL
        - reconstruction + LPIPS
        - reconstruction + KL + LPIPS
        - reconstruction + KL + LPIPS + GAN
        - optional feature matching

    Returns
    -------
    LossComposer
    """
    terms = []

    terms.append(
        _build_recon_term(
            recon_loss=recon_loss,
            pred_key=pred_key,
            target_key=target_key,
            weight=recon_weight,
        )
    )

    if use_kl:
        terms.append(
            KLLossTerm(
                posterior_key=posterior_key,
                weight=kl_weight,
                name="kl",
            )
        )

    if use_lpips:
        if lpips_model is None:
            raise ValueError("use_lpips=True requires lpips_model to be provided.")

        terms.append(
            LPIPSLossTerm(
                lpips_model=lpips_model,
                pred_key=pred_key,
                target_key=target_key,
                weight=lpips_weight,
                name="lpips",
                repeat_grayscale=repeat_grayscale_for_lpips,
                normalize_inputs=normalize_lpips_inputs,
            )
        )

    if use_adversarial:
        if discriminator is None:
            raise ValueError("use_adversarial=True requires discriminator to be provided.")

        terms.append(
            GeneratorAdversarialTerm(
                discriminator=discriminator,
                fake_key=pred_key,
                cond_key=cond_key,
                conditional=conditional,
                disc_start=disc_start,
                weight=adv_weight,
                name="g_adv",
                group="main",
            )
        )

        terms.append(
            DiscriminatorAdversarialTerm(
                discriminator=discriminator,
                real_key=target_key,
                fake_key=pred_key,
                cond_key=cond_key,
                conditional=conditional,
                disc_start=disc_start,
                weight=disc_weight,
                mode=disc_mode,
                name="d_adv",
                group="disc",
            )
        )

        if use_feature_matching:
            terms.append(
                FeatureMatchingLossTerm(
                    discriminator=discriminator,
                    real_key=target_key,
                    fake_key=pred_key,
                    cond_key=cond_key,
                    conditional=conditional,
                    weight=feature_matching_weight,
                    name="feature_matching",
                    group="main",
                )
            )

    return LossComposer(terms)


def build_ldm_autoencoder_loss(
    lpips_model: Optional[nn.Module] = None,
    discriminator: Optional[nn.Module] = None,
    recon_loss: str = "l1",
    recon_weight: float = 1.0,
    kl_weight: float = 1e-6,
    perceptual_weight: float = 1.0,
    adv_weight: float = 0.5,
    disc_weight: float = 1.0,
    disc_start: int = 50001,
    disc_mode: str = "hinge",
    repeat_grayscale_for_lpips: bool = True,
    normalize_lpips_inputs: bool = False,
    use_feature_matching: bool = False,
    feature_matching_weight: float = 0.0,
    conditional: bool = False,
    cond_key: Optional[str] = None,
    pred_key: str = "recon",
    target_key: str = "input",
    posterior_key: str = "posterior",
) -> LossComposer:
    """
    First packaged LDM-style autoencoder recipe for OMA.

    This is not yet the exact original CompVis adaptive-weight objective,
    but it captures the common structure:
        recon + KL + LPIPS + GAN (+ optional feature matching)
    """
    return build_autoencoder_loss(
        recon_loss=recon_loss,
        recon_weight=recon_weight,
        use_kl=True,
        kl_weight=kl_weight,
        use_lpips=lpips_model is not None,
        lpips_model=lpips_model,
        lpips_weight=perceptual_weight,
        repeat_grayscale_for_lpips=repeat_grayscale_for_lpips,
        normalize_lpips_inputs=normalize_lpips_inputs,
        use_adversarial=discriminator is not None,
        discriminator=discriminator,
        adv_weight=adv_weight,
        disc_weight=disc_weight,
        disc_start=disc_start,
        disc_mode=disc_mode,
        use_feature_matching=use_feature_matching,
        feature_matching_weight=feature_matching_weight,
        conditional=conditional,
        cond_key=cond_key,
        pred_key=pred_key,
        target_key=target_key,
        posterior_key=posterior_key,
    )


def build_ae_l1_loss(
    pred_key: str = "recon",
    target_key: str = "input",
) -> LossComposer:
    return build_autoencoder_loss(
        recon_loss="l1",
        use_kl=False,
        use_lpips=False,
        use_adversarial=False,
        pred_key=pred_key,
        target_key=target_key,
    )


def build_ae_l1_kl_loss(
    kl_weight: float = 1e-6,
    pred_key: str = "recon",
    target_key: str = "input",
    posterior_key: str = "posterior",
) -> LossComposer:
    return build_autoencoder_loss(
        recon_loss="l1",
        use_kl=True,
        kl_weight=kl_weight,
        use_lpips=False,
        use_adversarial=False,
        pred_key=pred_key,
        target_key=target_key,
        posterior_key=posterior_key,
    )


def build_ae_l2_kl_loss(
    kl_weight: float = 1e-6,
    pred_key: str = "recon",
    target_key: str = "input",
    posterior_key: str = "posterior",
) -> LossComposer:
    return build_autoencoder_loss(
        recon_loss="l2",
        use_kl=True,
        kl_weight=kl_weight,
        use_lpips=False,
        use_adversarial=False,
        pred_key=pred_key,
        target_key=target_key,
        posterior_key=posterior_key,
    )