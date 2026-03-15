from __future__ import annotations

import torch

from oma import Trainer
from oma.data.datasets.numpy_dataset import NumpyDataset
from oma.data.datamodule.datamodules import LSplitDataModule
from oma.methods.diffusion_bridge_translation import DiffusionBridgeTranslationMethod

from oma.models.backbones.ncsnpp import NCSNpp
from oma.models.diffusion.bridge import DiffusionBridge

from oma.evaluation.manager import EvaluatorManager
from oma.evaluation.evaluators import SaveImageEvaluator


def main() -> None:
    datamodule = LSplitDataModule(
        dataset_cls=NumpyDataset,
        dataset_kwargs={
            "data_dir": "/auto/data2/farslan/datasets/IXI",
            "source_modality": "T1",
            "target_modality": "T2",
            "image_size": 256,
            "padding": True,
            "norm": True,
        },
        train_dataloader_kwargs={
            "batch_size": 4,
            "shuffle": True,
            "drop_last": True,
            "num_workers": 4,
        },
        val_dataloader_kwargs={
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 4,
        },
        test_dataloader_kwargs={
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 4,
        },
    )

    # Quick sanity check
    datamodule.setup(stage="fit")
    datamodule.setup(stage="validate")
    batch = next(iter(datamodule.train_dataloader()))
    print("Batch keys:", batch.keys())
    print("Source shape:", batch["source"].shape)
    print("Target shape:", batch["target"].shape)

    generator_params = {
        'self_recursion': True,                # Whether to use self-consistent recursion
        'image_size': 256,     # Image size
        'z_emb_dim': 256 ,                     # Dimension of the latent embedding
        'ch_mult': [1, 1, 2, 2, 4, 4],         # Channel multipliers for each resolution
        'num_res_blocks': 2,                   # Number of residual blocks
        'attn_resolutions': [16],              # Resolutions to apply attention
        'dropout': 0.0,                        # Dropout rate
        'resamp_with_conv': True,              # Whether to use convolutional upsampling
        'conditional': True,                   # Whether to use condition on time embedding
        'fir': True,                           # Whether to use FIR filters
        'fir_kernel': [1, 3, 3, 1],            # FIR filter kernel
        'skip_rescale': True ,                 # Whether to skip rescaling the skip connection
        'resblock_type': 'biggan' ,              # Type of the residual block
        'progressive': 'none' ,                 # Whether to use progressive training
        'progressive_input': 'residual' ,        # Type of the input for the progressive training
        'embedding_type': 'positional' ,         # Embedding type
        'combine_method': 'sum' ,                # Method to combine the skip connection
        'fourier_scale': 16  ,                 # Fourier scale
        'nf': 64          ,                    # Number of filters
        'num_channels': 2 ,                    # Number of channels in the input
        'nz': 100   ,                          # Number of latent dimensions
        'n_mlp': 3  ,                         # Number of MLP layers
        'centered': True  ,                    # Whether to center the input
        'not_use_tanh': False ,
    }

    diffusion_params = {
        'n_steps': 10,                         # Number of diffusion steps
        'beta_start': 0.1,                     # Beta start value of the diffusion process
        'beta_end': 3.0,                       # Beta end value of the diffusion process
        'gamma': 1,                            # Gamma value that controls noise in the end-point of the bridge
        'n_recursions': 2,                     # Max number of recursions (R)
        'consistency_threshold': 0.01 
    }

    bridge_model = NCSNpp(**generator_params)
    diffusion = DiffusionBridge(**diffusion_params)

    evaluator_manager = EvaluatorManager(
        evaluators={
            "debug_images": SaveImageEvaluator(
                name="debug_images",
                max_samples=2,
                save_every_n_steps=50,
            )
        }
    )


    method = DiffusionBridgeTranslationMethod(
        bridge_model=bridge_model,
        diffusion=diffusion,
        lambda_rec_loss=1.0,
        optimizer_cfg={
            "class": torch.optim.Adam,
            "params": {
                "lr": 1e-4,
                "betas": (0.9, 0.999),
            },
        },
        save_hparams=False,
        evaluator_manager=evaluator_manager
    
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="cuda",
        devices=1,
        precision="32-true",
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=2,
        # limit_train_batches=5
    )

    trainer.fit(method, datamodule=datamodule)


if __name__ == "__main__":
    main()