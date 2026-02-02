"""
Model version registry.
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger("model_registry")


class ModelRegistry:
    """
    Tracks model versions and metadata.
    """

    def __init__(self, registry_dir: str = "models/metadata"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "registry.json"
        if not self.registry_file.exists():
            self.registry_file.write_text(json.dumps([], indent=2))

    def register(self, metadata: Dict) -> None:
        metadata["timestamp"] = datetime.utcnow().isoformat() + "Z"

        registry = json.loads(self.registry_file.read_text())
        registry.append(metadata)

        self.registry_file.write_text(json.dumps(registry, indent=2))
        logger.info("Registered model version %s", metadata["version"])

    def latest(self) -> Dict:
        registry = json.loads(self.registry_file.read_text())
        return registry[-1] if registry else {}
