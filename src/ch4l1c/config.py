from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

PERMIAN_BBOX = (-105.0, 29.5, -99.0, 34.0)

@dataclass(frozen=True)
class ProjectConfig:
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root / "data"
    carbon_mapper_dir: Path = data_dir / "carbon_mapper"
    emit_dir: Path = data_dir / "emit"
    label_dir: Path = data_dir / "labels"
    splits_dir: Path = data_dir / "raw" / "splits"
    training_dir: Path = data_dir / "training_l1c"
    model_dir: Path = data_dir / "models"
    output_dir: Path = data_dir / "outputs"


CFG = ProjectConfig()