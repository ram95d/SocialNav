"""CLI for training the MaskConvAutoencoder model."""

import os
import logging

from lightning.pytorch.cli import LightningCLI, ArgsType
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import model.model_utils as model_utils

# Import the model and data module classes, so that they can be used in the CLI.
from model.mask_autoenc.mask_autoencoder import MaskConvAutoencoderLitModule
from model.datasets.mask_dataset import MaskDataModule


# Logging setup.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaskLightningCLI(LightningCLI):
    """Custom LightningCLI that adds some additional functionality."""

    def add_arguments_to_parser(self, parser):

        # Debug argument.
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode."
        )

        # Add default logger.
        parser.set_defaults({
            "trainer.logger": {
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": "logs",
                    "default_hp_metric": False,
                },
            },
        })

        # Add default checkpoint callback.
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
        parser.set_defaults({"checkpoint_callback.save_top_k": 1,
                             "checkpoint_callback.monitor": "val_loss",
                             "checkpoint_callback.mode": "min",
                             "checkpoint_callback.save_last": True})

        # Add experiment name argument.
        parser.add_argument(
            "--experiment_name",
            default="unnamed_exp",
            type=str,
            help="Name of the experiment"
        )

        # Use experiment name in logger.
        parser.link_arguments("experiment_name",
                              "trainer.logger.init_args.name")

        # Checkpoint filename is a function of the experiment name.
        ckpt_name_fun = \
            lambda x: f"{x}" + "-{epoch:02d}-{val_loss:.4f}"
        parser.link_arguments("experiment_name",
                              "checkpoint_callback.filename",
                              compute_fn=ckpt_name_fun)

        # Add default learning rate monitor callback.
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")


def main(args: ArgsType = None):
    import warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*to avoid having duplicate data.*")

    # Log git info.
    sha, diff, branch = model_utils.git_info()
    logger.info(f"Git - sha={sha} branch={branch} diff='{diff}'")

    cli = MaskLightningCLI(args=args, run=False)

    try:
        cli.trainer.fit(cli.model, cli.datamodule)
        val_res = cli.trainer.validate(cli.model,
                                    cli.datamodule,
                                    ckpt_path='best')[0]
        test_res = cli.trainer.test(cli.model,
                                    cli.datamodule,
                                    ckpt_path='best')[0]

        result = {
            "val_loss": val_res["val_loss"],
            "test_loss": test_res["test_loss"],
        }

        model_utils.save_results(os.path.join(cli.trainer.log_dir,
                                              "results.yaml"),
                                 result)
    except Exception as e:
        if cli.config.debug:
            import traceback; traceback.print_exc()
            import pdb; pdb.post_mortem()
        else:
            raise e


if __name__ == "__main__":
    main()
