"""Episodic Memory Logger for the Edge Wake Phase.

Manages the short-term hippocampal buffer (`wake_buffer.jsonl`) on the local laptop
before asynchronous nocturnal consolidation on the HPC cluster.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import MemoryCategory, WakeEvent
from src.utils import append_jsonl, read_jsonl


class EpisodicMemoryBuffer:
    """Manages reading, appending, and archiving of daily WakeEvent records."""

    def __init__(
        self,
        buffer_path: str | Path = "data/wake_buffer.jsonl",
        archive_dir: str | Path = "data/archived_buffers",
    ) -> None:
        self.buffer_path = Path(buffer_path)
        self.archive_dir = Path(archive_dir)

    def record_event(
        self,
        fact: str,
        category: MemoryCategory = MemoryCategory.PROJECT_FACT,
        context: str | None = None,
        priority: int = 3,
        verification_question: str | None = None,
        expected_keywords: list[str] | None = None,
    ) -> WakeEvent:
        """Validate and persist a new episodic memory into the wake buffer."""
        event = WakeEvent(
            fact=fact,
            category=category,
            context=context,
            priority=priority,
            verification_question=verification_question,
            expected_keywords=expected_keywords or [],
        )
        append_jsonl(self.buffer_path, event)
        return event

    def load_events(self, min_priority: int = 1) -> list[WakeEvent]:
        """Load all validated events from the active wake buffer filtering by priority."""
        events = read_jsonl(self.buffer_path, WakeEvent)
        return [evt for evt in events if evt.priority >= min_priority]

    def build_episodic_prompt_context(self, max_events: int = 15) -> str:
        """Format recent high-priority wake events as an in-context memory block.

        During wakefulness, the model relies on this non-parametric buffer (hippocampus)
        until the sleep phase consolidates the facts into LoRA weights (neocortex).
        """
        events = self.load_events()
        if not events:
            return ""
        # Sort by priority (descending) then timestamp (most recent last)
        ranked = sorted(events, key=lambda e: (e.priority, e.timestamp))[-max_events:]
        lines = [
            "### Active Episodic Memory Buffer (Unconsolidated Session Facts):",
            *[f"- [{e.category.value.upper()} | P{e.priority}] {e.fact}" for e in ranked],
        ]
        return "\n".join(lines)

    def archive_and_clear(self, cycle_id: str | None = None) -> Path | None:
        """Archive the current wake buffer after a successful sleep cycle and reset it."""
        if not self.buffer_path.exists() or self.buffer_path.stat().st_size == 0:
            return None
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = cycle_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = self.archive_dir / f"wake_buffer_{stamp}.jsonl"
        shutil.move(str(self.buffer_path), str(dest))
        return dest
