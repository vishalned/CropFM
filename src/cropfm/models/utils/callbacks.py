from pytorch_lightning.callbacks import Callback
import torch
import psutil
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.loggers import WandbLogger


class RealTimeMemoryCallback(Callback):
    """
    Callback to log real-time memory usage during training
    Args:
        trainer: Trainer object
        pl_module: LightningModule object
        outputs: Outputs from the model
        batch: Batch of data
        batch_idx: Batch index
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                # GPU memory (same as nvidia-smi)
                free, total = torch.cuda.mem_get_info()
                mem_used_gb = (total - free) / (1024**3)
                pl_module.log("gpu_gb", mem_used_gb, prog_bar=True, logger=False)

            # CPU usage percentage
            cpu_usage = psutil.cpu_percent(interval=None)  # Non-blocking
            pl_module.log("cpu_%", cpu_usage, prog_bar=True, logger=False)

            # RAM usage in GB
            ram_used_gb = psutil.virtual_memory().used / (1024**3)
            pl_module.log("ram_gb", ram_used_gb, prog_bar=True, logger=False)


def configure_callbacks(cfg, model, pretraining=False):
    """
    Configure callbacks for training
    Args:
        cfg: Configuration object
        model: Model object
        pretraining: Whether the model is being trained for pretraining
    Returns:
        Dictionary with loggers and callbacks
    """
    callbacks_list = cfg.callbacks.callbacks_list
    configs = cfg.callbacks

    configurations = {"loggers": [], "callbacks": []}
    for callback in callbacks_list:
        match callback:
            case "wandb":
                config_dict = OmegaConf.to_container(cfg, resolve=True)
                configurations["loggers"].append(
                    WandbLogger(
                        project=configs.wandb.project,
                        name=configs.wandb.name,
                        config=config_dict,
                    )
                )
            case "checkpoint":
                configurations["callbacks"].append(
                    ModelCheckpoint(
                        monitor=(
                            f"val_{model.primary_metric_name}"
                            if not pretraining
                            else f"train_loss"
                        ),
                        mode=configs.checkpoint.mode,
                        save_top_k=configs.checkpoint.save_top_k,
                        save_last=configs.checkpoint.save_last,
                        dirpath=cfg.trainer.default_root_dir + "/model_checkpoints",
                        filename=configs.checkpoint.filename,
                    )
                )
            case "early_stopping":
                if pretraining:
                    raise ValueError("Early stopping is not supported for pretraining")
                configurations["callbacks"].append(
                    EarlyStopping(
                        monitor=f"val_{model.primary_metric_name}",
                        mode="max",
                        patience=10,
                        verbose=True,
                    )
                )
            case "real_time_memory":
                configurations["callbacks"].append(RealTimeMemoryCallback())
            case "model_summary":
                configurations["callbacks"].append(
                    ModelSummary(max_depth=configs.model_summary.max_depth)
                )

    if len(configurations["loggers"]) == 0:
        configurations["loggers"] = None
    if len(configurations["callbacks"]) == 0:
        configurations["callbacks"] = None

    return configurations