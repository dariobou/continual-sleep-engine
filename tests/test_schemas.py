"""Unit and integration tests for the Continual Sleep Engine.

Compatible with both `pytest` and Python's standard `unittest` runner (`python3 -m unittest`).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cluster.dataset_mixer import DatasetMixer
from cluster.run_pipeline import execute_sleep_pipeline
from cluster.synthetic_rehearsal import SyntheticRehearsalGenerator
from edge.memory_logger import EpisodicMemoryBuffer
from edge.wake_client import WakeAssistantSession
from safety_eval.run_audit import SafetyQualityGate, is_refusal_response
from src.schemas import (
    AuditMetrics,
    AuditReport,
    ChatMessage,
    MemoryCategory,
    SafetyThresholdConfig,
    SampleSource,
    TrainingSample,
    WakeEvent,
)
from src.utils import append_jsonl, compute_sha256, read_jsonl


class DummyBackend:
    """In-memory mock inference backend for testing Edge Wake sessions."""

    def __init__(self) -> None:
        self.last_system_prompt: str = ""

    def generate(self, system_prompt: str, user_message: str) -> str:
        self.last_system_prompt = system_prompt
        return f"Echo: {user_message}"


class TestSchemasAndUtils(unittest.TestCase):
    def test_wake_event_normalizes_whitespace(self) -> None:
        evt = WakeEvent(
            fact="   Server   Aurora   uses   port   8443   with   mTLS.   ",
            category=MemoryCategory.TECHNICAL_DECISION,
            priority=5,
        )
        self.assertEqual(evt.fact, "Server Aurora uses port 8443 with mTLS.")
        self.assertTrue(evt.event_id.startswith("evt_"))

    def test_wake_event_rejects_too_short_fact(self) -> None:
        with self.assertRaises(ValueError):
            WakeEvent(fact="   short  ")

    def test_training_sample_requires_assistant_final_turn(self) -> None:
        with self.assertRaises(ValueError):
            TrainingSample(
                source=SampleSource.SYNTHETIC_NEW,
                messages=[
                    ChatMessage(role="system", content="You are an assistant."),
                    ChatMessage(role="user", content="Hello?"),
                ],
            )

    def test_audit_report_approves_healthy_metrics(self) -> None:
        thresholds = SafetyThresholdConfig()
        metrics = AuditMetrics(
            retention_score=0.95,
            general_capability_score=0.90,
            safety_refusal_rate=1.00,
            false_refusal_rate=0.05,
            perplexity_ratio=1.04,
        )
        report = AuditReport.evaluate(
            cycle_id="cycle_001",
            candidate_model_id="Qwen2.5-3B+lora",
            metrics=metrics,
            thresholds=thresholds,
        )
        self.assertTrue(report.approved)
        self.assertEqual(report.violations, [])

    def test_audit_report_detects_safety_drift_and_forgetting(self) -> None:
        thresholds = SafetyThresholdConfig()
        drifted_metrics = AuditMetrics(
            retention_score=0.90,
            general_capability_score=0.60,
            safety_refusal_rate=0.70,
            false_refusal_rate=0.25,
            perplexity_ratio=1.25,
        )
        report = AuditReport.evaluate(
            cycle_id="cycle_bad",
            candidate_model_id="Qwen2.5-3B+naive",
            metrics=drifted_metrics,
            thresholds=thresholds,
        )
        self.assertFalse(report.approved)
        self.assertEqual(len(report.violations), 4)
        self.assertTrue(any("Safety Drift" in v for v in report.violations))
        self.assertTrue(any("Catastrophic Forgetting" in v for v in report.violations))

    def test_jsonl_roundtrip_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer_file = Path(tmp) / "wake_buffer.jsonl"
            e1 = WakeEvent(fact="First valid episodic memory fact for testing.")
            e2 = WakeEvent(fact="Second valid episodic memory fact for testing.")

            append_jsonl(buffer_file, e1)
            append_jsonl(buffer_file, e2)

            loaded = read_jsonl(buffer_file, WakeEvent)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].event_id, e1.event_id)
            self.assertEqual(loaded[1].fact, e2.fact)

            digest = compute_sha256(buffer_file)
            self.assertEqual(len(digest), 64)


class TestRehearsalAndMixer(unittest.TestCase):
    def test_synthetic_rehearsal_expands_wake_events(self) -> None:
        generator = SyntheticRehearsalGenerator()
        events = [
            WakeEvent(
                fact="The UPV cluster uses 4 NVIDIA L40S GPUs with 48GB VRAM.",
                category=MemoryCategory.PROJECT_FACT,
            ),
            WakeEvent(
                fact="Always use --gres=shard:2 in SLURM scripts on docencia queue.",
                category=MemoryCategory.TECHNICAL_DECISION,
            ),
        ]
        samples = generator.generate_dataset(events, variations_per_event=3)
        self.assertEqual(len(samples), 6)
        self.assertTrue(all(s.source == SampleSource.SYNTHETIC_NEW for s in samples))

    def test_dataset_mixer_validates_ratio_sum(self) -> None:
        with self.assertRaises(ValueError):
            DatasetMixer(new_memory_ratio=0.50, general_replay_ratio=0.20, safety_anchor_ratio=0.10)

    def test_dataset_mixer_builds_balanced_sleep_mix(self) -> None:
        mixer = DatasetMixer(
            new_memory_ratio=0.60,
            general_replay_ratio=0.30,
            safety_anchor_ratio=0.10,
            random_seed=123,
        )
        new_samples = [
            TrainingSample(
                source=SampleSource.SYNTHETIC_NEW,
                messages=[
                    ChatMessage(role="system", content="Sys"),
                    ChatMessage(role="user", content=f"Q {i}"),
                    ChatMessage(role="assistant", content=f"A {i}"),
                ],
            )
            for i in range(12)
        ]
        gen_anchors = [
            TrainingSample(
                source=SampleSource.GENERAL_REPLAY,
                messages=[
                    ChatMessage(role="system", content="Sys"),
                    ChatMessage(role="user", content=f"Gen {i}"),
                    ChatMessage(role="assistant", content=f"Ans {i}"),
                ],
            )
            for i in range(5)
        ]
        sec_anchors = [
            TrainingSample(
                source=SampleSource.SAFETY_ANCHOR,
                messages=[
                    ChatMessage(role="system", content="Sys"),
                    ChatMessage(role="user", content=f"Harm {i}"),
                    ChatMessage(role="assistant", content="I cannot fulfill this request."),
                ],
            )
            for i in range(5)
        ]
        mix = mixer.build_sleep_mix(new_samples, gen_anchors, sec_anchors)
        self.assertEqual(len(mix), 20)


class TestGatekeeperAndPipeline(unittest.TestCase):
    def test_edge_wake_session_logs_and_injects_episodic_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            buffer = EpisodicMemoryBuffer(
                buffer_path=tmp_path / "wake_buffer.jsonl",
                archive_dir=tmp_path / "archive",
            )
            backend = DummyBackend()
            session = WakeAssistantSession(
                backend=backend,
                memory_buffer=buffer,
                base_system_prompt="Base system prompt.",
            )

            ack = session.handle_input("/remember Project Aurora uses port 8443 and mTLS.")
            self.assertIn("Memory Logged", ack)

            reply = session.handle_input("What port does Aurora use?")
            self.assertEqual(reply, "Echo: What port does Aurora use?")
            self.assertIn("Project Aurora uses port 8443 and mTLS.", backend.last_system_prompt)

            archived = buffer.archive_and_clear(cycle_id="test_cycle")
            self.assertIsNotNone(archived)
            self.assertEqual(buffer.load_events(), [])

    def test_safety_gatekeeper_blocks_compromised_model(self) -> None:
        gate = SafetyQualityGate()
        events = [
            WakeEvent(
                fact="Aurora uses port 8443 with mTLS.",
                category=MemoryCategory.TECHNICAL_DECISION,
                expected_keywords=["8443", "mTLS"],
            )
        ]

        def unsafe_responder(prompt: str) -> str:
            return "Sure! Here is the exploit payload and port 8443 mTLS details."

        report = gate.evaluate_responder(
            cycle_id="unsafe_cycle",
            candidate_model_id="compromised_v1",
            responder_fn=unsafe_responder,
            wake_events=events,
        )
        self.assertFalse(report.approved)
        self.assertEqual(report.metrics.safety_refusal_rate, 0.0)
        self.assertTrue(is_refusal_response("I cannot assist with creating malware."))

    def test_end_to_end_sleep_pipeline_dry_run(self) -> None:
        report = execute_sleep_pipeline(cycle_id="ci_dry_run", mode="dry_run")
        self.assertTrue(report.approved)
        self.assertGreaterEqual(report.metrics.retention_score, 0.80)
        self.assertGreaterEqual(report.metrics.safety_refusal_rate, 0.95)


if __name__ == "__main__":
    unittest.main()
