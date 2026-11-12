"""Nocturnal LoRA Consolidation & Weight Merging Pipeline.

Supports two execution modes:
1. `dry_run` (Local CPU/WSL2 validation): Verifies dataset formatting, LoRA config
   compatibility, and emits a simulated candidate artifact manifest.
2. `hpc_gpu` (UPV tensor.dsic.upv.es L40S): Loads the FP16/BF16 base checkpoint,
   trains the low-rank adapter (W_new = W_0 + B*A), executes `merge_and_unload()`,
   and saves the high-precision merged model ready for GGUF quantization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.schemas import TrainingSample
from src.utils import compute_sha256, load_yaml_config, read_jsonl


@dataclass
class ConsolidationResult:
    """Summary returned after LoRA training and weight fusion."""

    cycle_id: str
    strategy: Literal["sequential", "cumulative"]
    base_model_id: str
    num_training_samples: int
    merged_checkpoint_dir: Path
    simulated: bool
    manifest_sha256: str


class LoRAConsolidator:
    """Manages PEFT/LoRA training and mathematical weight merging."""

    def __init__(self, lora_config_path: str | Path = "configs/lora_config.yaml") -> None:
        self.config = load_yaml_config(lora_config_path)
        self.model_cfg = self.config.get("model", {})
        self.lora_cfg = self.config.get("lora", {})
        self.train_cfg = self.config.get("training", {})

    def validate_lora_hyperparameters(self) -> None:
        """Sanity-check LoRA rank, alpha, and target transformer modules."""
        r = self.lora_cfg.get("r", 0)
        alpha = self.lora_cfg.get("lora_alpha", 0)
        targets = self.lora_cfg.get("target_modules", [])
        if r <= 0 or alpha <= 0:
            raise ValueError(f"LoRA rank (r={r}) and alpha ({alpha}) must be positive integers.")
        if not targets:
            raise ValueError("LoRA target_modules list cannot be empty.")

    def consolidate(
        self,
        cycle_id: str,
        train_jsonl_path: str | Path,
        output_dir: str | Path = "data/merged_models",
        dry_run: bool = True,
    ) -> ConsolidationResult:
        """Run LoRA fine-tuning and weight fusion (`merge_and_unload`)."""
        self.validate_lora_hyperparameters()
        samples = read_jsonl(train_jsonl_path, TrainingSample)
        if not samples:
            raise ValueError(f"No valid TrainingSamples found in {train_jsonl_path}")

        strategy: Literal["sequential", "cumulative"] = self.model_cfg.get(
            "consolidation_strategy", "sequential"
        )
        base_model_id = self.model_cfg.get("base_model_id", "Qwen/Qwen2.5-3B-Instruct")
        target_dir = Path(output_dir) / f"checkpoint_{cycle_id}"
        target_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            manifest = {
                "cycle_id": cycle_id,
                "mode": "dry_run_simulation",
                "base_model_id": base_model_id,
                "strategy": strategy,
                "lora_r": self.lora_cfg["r"],
                "lora_alpha": self.lora_cfg["lora_alpha"],
                "target_modules": self.lora_cfg["target_modules"],
                "num_samples": len(samples),
                "source_breakdown": {
                    src: sum(1 for s in samples if s.source.value == src)
                    for src in {"synthetic_new", "general_replay", "safety_anchor"}
                },
            }
            manifest_path = target_dir / "consolidation_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            digest = compute_sha256(manifest_path)
            return ConsolidationResult(
                cycle_id=cycle_id,
                strategy=strategy,
                base_model_id=base_model_id,
                num_training_samples=len(samples),
                merged_checkpoint_dir=target_dir,
                simulated=True,
                manifest_sha256=digest,
            )

        # Full HPC GPU execution path (requires torch, transformers, peft, trl on L40S node)
        return self._run_hpc_gpu_training(
            cycle_id=cycle_id,
            samples=samples,
            base_model_id=base_model_id,
            strategy=strategy,
            target_dir=target_dir,
        )

    def _run_hpc_gpu_training(
        self,
        cycle_id: str,
        samples: list[TrainingSample],
        base_model_id: str,
        strategy: Literal["sequential", "cumulative"],
        target_dir: Path,
    ) -> ConsolidationResult:
        """Execute real PEFT LoRA training and FP16/BF16 merge on NVIDIA L40S GPU."""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer

        dtype = torch.bfloat16 if self.model_cfg.get("torch_dtype") == "bfloat16" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
        )

        peft_config = LoraConfig(
            r=self.lora_cfg["r"],
            lora_alpha=self.lora_cfg["lora_alpha"],
            lora_dropout=self.lora_cfg.get("lora_dropout", 0.05),
            bias=self.lora_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
            target_modules=self.lora_cfg["target_modules"],
        )
        peft_model = get_peft_model(model, peft_config)

        formatted_texts = [
            tokenizer.apply_chat_template(
                [m.model_dump() for m in sample.messages],
                tokenize=False,
            )
            for sample in samples
        ]
        hf_dataset = Dataset.from_dict({"text": formatted_texts})

        training_args = TrainingArguments(
            output_dir=str(target_dir / "adapter_tmp"),
            num_train_epochs=self.train_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=self.train_cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=self.train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=float(self.train_cfg.get("learning_rate", 2e-4)),
            bf16=(dtype == torch.bfloat16),
            fp16=(dtype == torch.float16),
            logging_steps=self.train_cfg.get("logging_steps", 5),
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=self.train_cfg.get("max_seq_length", 1024),
            tokenizer=tokenizer,
            args=training_args,
        )
        trainer.train()

        # Mathematical weight fusion: W_merged = W_base + (alpha / r) * (B @ A)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(str(target_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(target_dir))

        manifest_path = target_dir / "consolidation_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "cycle_id": cycle_id,
                    "mode": "hpc_gpu_merged",
                    "base_model_id": base_model_id,
                    "strategy": strategy,
                    "num_samples": len(samples),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        digest = compute_sha256(manifest_path)
        return ConsolidationResult(
            cycle_id=cycle_id,
            strategy=strategy,
            base_model_id=base_model_id,
            num_training_samples=len(samples),
            merged_checkpoint_dir=target_dir,
            simulated=False,
            manifest_sha256=digest,
        )
