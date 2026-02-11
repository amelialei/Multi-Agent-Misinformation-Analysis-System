from pathlib import Path


def load_prompt(agent_dir, filename="fcot.txt"):
    path = agent_dir / "prompts" / filename
    return path.read_text(encoding="utf-8")
