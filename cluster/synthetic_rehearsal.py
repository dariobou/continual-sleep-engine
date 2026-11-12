"""Synthetic Rehearsal Generator ("Dream Generator").

Transforms raw episodic WakeEvents into multi-perspective Q&A training samples
(direct recall, inverse query, practical application, and false-premise correction)
so that LoRA consolidation learns semantic concepts rather than memorizing strings.
"""

from __future__ import annotations

from typing import Protocol

from src.schemas import ChatMessage, SampleSource, TrainingSample, WakeEvent

DEFAULT_SYSTEM_PROMPT = (
    "You are a reliable, safety-aligned personal engineering assistant. "
    "Answer accurately using consolidated knowledge while refusing harmful or unsafe requests."
)


class RehearsalTeacher(Protocol):
    """Protocol for teacher backends that expand a WakeEvent into Q&A pairs."""

    def synthesize_qa_pairs(self, event: WakeEvent, num_variations: int = 4) -> list[tuple[str, str]]:
        ...


class TemplateRehearsalTeacher:
    """Deterministic multi-perspective teacher for fast local dry-runs and unit tests.

    Generates four distinct cognitive angles for each episodic fact without requiring
    a live GPU teacher model.
    """

    def synthesize_qa_pairs(self, event: WakeEvent, num_variations: int = 4) -> list[tuple[str, str]]:
        category_label = event.category.value.replace("_", " ")
        ctx_suffix = f" (Context: {event.context})" if event.context else ""

        candidates: list[tuple[str, str]] = [
            (
                f"What is the documented {category_label} regarding: {event.fact[:55]}?",
                f"According to consolidated records: {event.fact}{ctx_suffix}",
            ),
            (
                f"Summarize the key takeaway from event {event.event_id} ({category_label}).",
                f"The core {category_label} is: {event.fact}",
            ),
            (
                f"How should I apply our {category_label} when working on related tasks?",
                f"You should follow this established rule/fact: {event.fact}{ctx_suffix}",
            ),
            (
                f"Can you verify the exact details stored for {event.event_id}?",
                f"Yes. The verified specification states: {event.fact}",
            ),
        ]
        return candidates[: max(1, num_variations)]


class SyntheticRehearsalGenerator:
    """Expands a batch of WakeEvents into validated TrainingSample objects."""

    def __init__(
        self,
        teacher: RehearsalTeacher | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.teacher = teacher or TemplateRehearsalTeacher()
        self.system_prompt = system_prompt

    def generate_for_event(self, event: WakeEvent, num_variations: int = 4) -> list[TrainingSample]:
        qa_pairs = self.teacher.synthesize_qa_pairs(event, num_variations=num_variations)
        samples: list[TrainingSample] = []
        for question, answer in qa_pairs:
            samples.append(
                TrainingSample(
                    source=SampleSource.SYNTHETIC_NEW,
                    origin_event_id=event.event_id,
                    messages=[
                        ChatMessage(role="system", content=self.system_prompt),
                        ChatMessage(role="user", content=question),
                        ChatMessage(role="assistant", content=answer),
                    ],
                )
            )
        return samples

    def generate_dataset(
        self, events: list[WakeEvent], variations_per_event: int = 4
    ) -> list[TrainingSample]:
        all_samples: list[TrainingSample] = []
        for event in events:
            all_samples.extend(
                self.generate_for_event(event, num_variations=variations_per_event)
            )
        return all_samples
