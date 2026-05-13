"""
SAM3 LoRA Fine-tuning for Remote Sensing Segmentation
======================================================

This script applies LoRA adapters to SAM3 and fine-tunes on remote sensing datasets.
Memory efficient and maintains pre-trained knowledge while adapting to domain.

Usage:
    python train_sam3_lora_remote_sensing.py \
        --config hsi_lora_finetuning.yaml \
        --num-gpus 1 \
        --use-lora \
        --lora-rank 16
"""

import logging
import os
import sys
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from hydra import initialize, compose, initialize_config_module

from hydra.utils import instantiate
from omegaconf import OmegaConf
from peft import get_peft_model, LoraConfig, TaskType

torch.cuda.init() 
from hydra.core.global_hydra import GlobalHydra

from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers
from sam3.train.utils.distributed import unwrap_ddp_if_wrapped
from iopath.common.file_io import g_pathmgr
import torch.distributed as dist
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HYDRA_FULL_ERROR"] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"DEVICE {device}")

import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.original = original
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Linear(original.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False
        # Initialize: A ~ N(0,1), B = 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Move LoRA weights to same device as original
        self.lora_A = self.lora_A.to(original.weight.device)
        self.lora_B = self.lora_B.to(original.weight.device)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def sam3_list_wrapper(batch, collate_fn):
    # Calls the actual collator, then wraps the result in a list
    return [collate_fn(batch)]

# def apply_lora_to_model(model, lora_config):
#     """
#     Apply LoRA adapters to the model.
    
#     Args:
#         model: SAM3 model
#         lora_config: LoRA configuration dict
    
#     Returns:
#         Model with LoRA adapters applied
#     """
#     if not lora_config.get('enabled', False):
#         logger.info("LoRA is disabled in config")
#         return model
    
#     logger.info("="*80)
#     logger.info("APPLYING LORA ADAPTERS TO SAM3")
#     logger.info("="*80)
    
#     # LoRA configuration
#     peft_config = LoraConfig(
#         r=lora_config['rank'],
#         lora_alpha=lora_config['lora_alpha'],
#         target_modules=lora_config['target_modules'],
#         lora_dropout=lora_config['lora_dropout'],
#         bias=lora_config['bias'],
#         task_type=None,  # For feature extraction
#     )
    
#     # Apply LoRA
#     model = get_peft_model(model, peft_config)
    
#     # Print trainable parameters
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
    
#     logger.info(f"\nLoRA Applied Successfully!")
#     logger.info(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
#     logger.info(f"  Total parameters:     {total_params:,}")
#     logger.info(f"  Frozen parameters:    {total_params - trainable_params:,}")
    
#     # Print LoRA details
#     logger.info(f"\n  LoRA Rank:            {lora_config['rank']}")
#     logger.info(f"  LoRA Alpha:           {lora_config['lora_alpha']}")
#     logger.info(f"  LoRA Dropout:         {lora_config['lora_dropout']}")
#     logger.info(f"  Target Modules:       {lora_config['target_modules']}")
    
#     model.print_trainable_parameters()
    
#     return model

def apply_lora_to_model(model, lora_config):
    if not lora_config.get('enabled', False):
        logger.info("LoRA is disabled in config")
        return model

    import math
    r = lora_config['rank']
    alpha = lora_config['lora_alpha']
    dropout = lora_config['lora_dropout']
    target_modules = list(lora_config['target_modules'])  # e.g. ["attn.proj"]

    logger.info("="*80)
    logger.info("APPLYING MANUAL LORA ADAPTERS TO SAM3")
    logger.info("="*80)

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Navigate to parent and replace
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                child_name = parts[-1]
                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                replaced += 1
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Replaced {replaced} linear layers with LoRA")
    logger.info(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    logger.info(f"Total:     {total:,}")

    return model  # returns plain nn.Module, no PEFT wrapper

def print_training_info(cfg):
    """Print comprehensive training information."""
    print("\n" + "="*80)
    print("SAM3 LORA FINE-TUNING - REMOTE SENSING SEGMENTATION")
    print("="*80)
    
    print("\n📊 DATASET CONFIGURATION:")
    print(f"  Training images:      {cfg.dataset.train.img_folder}")
    print(f"  Validation images:    {cfg.dataset.val.img_folder}")
    print(f"  Resolution:           {cfg.scratch.resolution}x{cfg.scratch.resolution}")
    print(f"  Enable segmentation:  {cfg.scratch.enable_segmentation}")
    
    print("\n⚙️  LORA CONFIGURATION:")
    print(f"  Enabled:              {cfg.lora_config.enabled}")
    print(f"  Rank:                 {cfg.lora_config.rank}")
    print(f"  Alpha:                {cfg.lora_config.lora_alpha}")
    print(f"  Dropout:              {cfg.lora_config.lora_dropout}")
    print(f"  Target modules:       {cfg.lora_config.target_modules}")
    
    print("\n🎓 TRAINING CONFIGURATION:")
    print(f"  Batch size:           {cfg.scratch.train_batch_size} (per GPU)")
    print(f"  Gradient accumulation: {cfg.scratch.gradient_accumulation_steps}")
    print(f"  Effective batch:      {cfg.scratch.train_batch_size * cfg.scratch.gradient_accumulation_steps}")
    print(f"  Max epochs:           {cfg.scratch.max_epochs}")
    print(f"  Mixed precision:      {cfg.trainer.optim.amp.amp_dtype}")
    print(f"  Activation ckpt:      {cfg.scratch.use_activation_checkpointing}")
    
    print("\n📈 LEARNING RATES:")
    print(f"  Transformer:          {cfg.scratch.lr_transformer}")
    print(f"  Vision backbone:      {cfg.scratch.lr_vision_backbone}")
    print(f"  Language backbone:    {cfg.scratch.lr_language_backbone}")
    print(f"  Weight decay:         {cfg.scratch.wd}")
    
    print("\n💾 CHECKPOINT & LOGGING:")
    print(f"  Save directory:       {cfg.trainer.checkpoint.save_dir}")
    print(f"  Log directory:        {cfg.trainer.logging.log_dir}")
    print(f"  Tensorboard:          {cfg.trainer.logging.tensorboard_writer.log_dir}")
    
    print("\n" + "="*80)


def setup_and_train(cfg, num_gpus=1):
    """Setup training with LoRA and run."""
    logger.info(f"Setting up training with {num_gpus} GPU(s)")


    
    # Set distributed environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 65000))
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(num_gpus)
    
    # Register resolvers
    register_omegaconf_resolvers()
    
    # Create log directory
    makedir(cfg.launcher.experiment_log_dir)
    
    # Save config
    config_path = os.path.join(cfg.launcher.experiment_log_dir, "config.yaml")
    with g_pathmgr.open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")
    
    # Print training info
    print_training_info(cfg)
    
    # Instantiate trainer
    logger.info("Building trainer...")

    # Build trainer normally (it instantiates the model internally)
    trainer = instantiate(cfg.trainer, _recursive_=False)

    # Apply LoRA to the already-built model (unwrap DDP first if needed)
    raw_model = unwrap_ddp_if_wrapped(trainer.model)
    raw_model = apply_lora_to_model(raw_model, cfg.lora_config)
    trainer.model = raw_model

    # Rebuild optimizers so they see only the LoRA parameters
    trainer._construct_optimizers()


    # print("\nMODEL BEFORE LORA")
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total = sum(p.numel() for p in model.parameters())
    # print(trainable, total)

    # model = apply_lora_to_model(model, cfg.lora_config)

    # print("\nMODEL AFTER LORA")
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total = sum(p.numel() for p in model.parameters())
    # print(trainable, total)

    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name)
    
    # Run training
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80 + "\n")
    
    try:
        trainer.run()
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {cfg.launcher.experiment_log_dir}")
        logger.info("="*80)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main(args):
    """Main entry point."""
    logger.info(f"Loading config: {args.config}")
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(config_path="hsi_dataset", version_base=None):
        cfg = compose(config_name=args.config)
    
    # Override GPU count
    if args.num_gpus:
        cfg.launcher.gpus_per_node = args.num_gpus
        logger.info(f"Overriding GPUs per node to {args.num_gpus}")
    
    # Override LoRA settings if provided
    if args.lora_rank:
        cfg.lora_config.rank = args.lora_rank
        logger.info(f"Overriding LoRA rank to {args.lora_rank}")
    
    if args.no_lora:
        cfg.lora_config.enabled = False
        logger.info("LoRA disabled via command line")
    
    # Run setup and training
    try:
        # Run setup and training
        setup_and_train(cfg, num_gpus=cfg.launcher.gpus_per_node)
    finally:
        # This ALWAYS runs, even if training fails
        if dist.is_initialized():
            logger.info("Cleaning up distributed process group...")
            dist.destroy_process_group()


if __name__ == "__main__":
    #initialize_config_module("sam3.train", version_base="1.2")
    
    parser = ArgumentParser(description="SAM3 LoRA Fine-tuning for Remote Sensing")
    parser.add_argument(
        "-c", "--config",
        default="hsi_train",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank (overrides config)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA",
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)