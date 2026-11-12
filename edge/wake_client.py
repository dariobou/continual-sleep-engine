"""Edge Wake Phase Client for local SLM inference (RTX 3060 6GB via Ollama).

Combines fast 4-bit GGUF inference with real-time episodic memory injection
and explicit `/remember` fact capture.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Protocol

from edge.memory_logger import EpisodicMemoryBuffer
from src.schemas import MemoryCategory
from src.utils import load_yaml_config


class InferenceBackend(Protocol):
    """Protocol for swappable local inference engines (Ollama, llama.cpp, or Mock)."""

    def generate(self, system_prompt: str, user_message: str) -> str:
        ...


class OllamaBackend:
    """HTTP client adapter for local Ollama server running quantized GGUF models."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        timeout: float = 60.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_message: str) -> str:
        url = f"{self.host}/api/chat"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"]


class WakeAssistantSession:
    """Coordinates local inference with the hippocampal episodic buffer."""

    def __init__(
        self,
        backend: InferenceBackend,
        memory_buffer: EpisodicMemoryBuffer,
        base_system_prompt: str,
        max_episodic_events: int = 15,
    ) -> None:
        self.backend = backend
        self.memory_buffer = memory_buffer
        self.base_system_prompt = base_system_prompt
        self.max_episodic_events = max_episodic_events

    def build_effective_system_prompt(self) -> str:
        episodic_block = self.memory_buffer.build_episodic_prompt_context(
            max_events=self.max_episodic_events
        )
        if not episodic_block:
            return self.base_system_prompt
        return f"{self.base_system_prompt}\n\n{episodic_block}"

    def handle_input(self, user_input: str) -> str:
        """Process either a `/remember` command or a standard conversational turn."""
        cleaned = user_input.strip()
        if cleaned.startswith("/remember "):
            fact_text = cleaned[len("/remember ") :].strip()
            event = self.memory_buffer.record_event(
                fact=fact_text,
                category=MemoryCategory.PROJECT_FACT,
                priority=4,
            )
            return f"[Memory Logged: {event.event_id}] Stored for nocturnal consolidation: '{event.fact}'"

        system_prompt = self.build_effective_system_prompt()
        return self.backend.generate(system_prompt=system_prompt, user_message=cleaned)


def run_interactive_cli(config_path: str = "configs/engine_config.yaml") -> None:
    """Launch interactive terminal session for the Edge Wake Phase."""
    cfg = load_yaml_config(config_path)
    edge_cfg = cfg.get("edge", {})
    proj_cfg = cfg.get("project", {})

    buffer = EpisodicMemoryBuffer(
        buffer_path=edge_cfg.get("wake_buffer_path", "data/wake_buffer.jsonl"),
        archive_dir=edge_cfg.get("archive_dir", "data/archived_buffers"),
    )
    backend = OllamaBackend(
        host=edge_cfg.get("ollama_host", "http://localhost:11434"),
        model=edge_cfg.get("active_model_tag", "qwen2.5:3b-instruct-q4_K_M"),
    )
    session = WakeAssistantSession(
        backend=backend,
        memory_buffer=buffer,
        base_system_prompt=proj_cfg.get("default_system_prompt", "You are a helpful assistant."),
        max_episodic_events=edge_cfg.get("max_episodic_context_events", 15),
    )

    print("=" * 68)
    print("Continual Sleep Engine — Wake Phase Client (Edge Node RTX 3060 6GB)")
    print("Commands: /remember <fact> | /buffer | /sleep")
    print("=" * 68)

    while True:
        try:
            user_text = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            continue
        if user_text in {"/sleep", "/exit", "exit", "quit"}:
            print("Closing Wake Phase. Ready for nocturnal sync.")
            break
        if user_text == "/buffer":
            events = buffer.load_events()
            print(f"Unconsolidated events in buffer: {len(events)}")
            for evt in events:
                print(f"  • ({evt.event_id}) [{evt.category.value}] {evt.fact}")
            continue

        try:
            reply = session.handle_input(user_text)
            print(f"\nAssistant > {reply}")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Wake Phase CLI")
    parser.add_argument("--config", default="configs/engine_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run_interactive_cli(args.config)
