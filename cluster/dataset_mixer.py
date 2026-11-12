"""Complementary Learning Systems (CLS) Dataset Mixer.

Combines newly synthesized dream memories (60%) with general capability replay
anchors (30%) and explicit safety refusal anchors (10%) to prevent both
Catastrophic Forgetting and Safety Drift during nocturnal LoRA fine-tuning.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from src.schemas import ChatMessage, SampleSource, TrainingSample


class DatasetMixer:
    """Balances new episodic memories, general replay, and safety alignment anchors."""

    def __init__(
        self,
        new_memory_ratio: float = 0.60,
        general_replay_ratio: float = 0.30,
        safety_anchor_ratio: float = 0.10,
        random_seed: int = 42,
    ) -> None:
        total = new_memory_ratio + general_replay_ratio + safety_anchor_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-3):
            raise ValueError(
                f"Mixer ratios must sum to 1.0 (got {total:.4f}: "
                f"new={new_memory_ratio}, replay={general_replay_ratio}, safety={safety_anchor_ratio})"
            )
        self.new_memory_ratio = new_memory_ratio
        self.general_replay_ratio = general_replay_ratio
        self.safety_anchor_ratio = safety_anchor_ratio
        self.random_seed = random_seed

    @staticmethod
    def _sample_with_replacement(
        pool: list[TrainingSample], count: int, rng: random.Random
    ) -> list[TrainingSample]:
        if count <= 0 or not pool:
            return []
        if count <= len(pool):
            return rng.sample(pool, count)
        # If more samples are needed than exist in the anchor pool, cycle + shuffle
        selected: list[TrainingSample] = []
        while len(selected) < count:
            remaining = count - len(selected)
            batch = rng.sample(pool, min(remaining, len(pool)))
            selected.extend(batch)
        return selected

    def build_sleep_mix(
        self,
        synthetic_new: list[TrainingSample],
        general_anchors: list[TrainingSample],
        safety_anchors: list[TrainingSample],
    ) -> list[TrainingSample]:
        """Construct a deterministically shuffled training mix following target ratios."""
        if not synthetic_new:
            raise ValueError("Cannot build sleep mix: 'synthetic_new' sample list is empty.")

        rng = random.Random(self.random_seed)
        n_new = len(synthetic_new)

        # Compute total target size from the new memory count and its ratio
        total_target = max(n_new, round(n_new / self.new_memory_ratio))
        n_general = max(1, round(total_target * self.general_replay_ratio)) if general_anchors else 0
        n_safety = max(1, round(total_target * self.safety_anchor_ratio)) if safety_anchors else 0

        sampled_general = self._sample_with_replacement(general_anchors, n_general, rng)
        sampled_safety = self._sample_with_replacement(safety_anchors, n_safety, rng)

        combined = list(synthetic_new) + sampled_general + sampled_safety
        rng.shuffle(combined)
        return combined

    @staticmethod
    def load_anchor_file(
        path: str | Path,
        source_type: SampleSource,
        system_prompt: str,
    ) -> list[TrainingSample]:
        """Load benchmark/anchor JSON array of {'prompt', 'expected_response'} into TrainingSamples."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Anchor file not found: {file_path}")
        raw_items = json.loads(file_path.read_text(encoding="utf-8"))
        samples: list[TrainingSample] = []
        for item in raw_items:
            samples.append(
                TrainingSample(
                    sample_id=item.get("id", f"anc_{source_type.value}"),
                    source=source_type,
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=item["prompt"]),
                        ChatMessage(role="assistant", content=item["reference_response"]),
                    ],
                )
            )
        return samples
