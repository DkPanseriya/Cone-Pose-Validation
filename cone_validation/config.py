"""Validation settings, loaded from config.yaml."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    names: list           # model class id -> short name
    fsoco_to_class: dict  # FSOCO class title -> model class id
    heights_m: list       # model class id -> cone height
    ignore_classes: set
    keypoints: list
    gate_enabled: bool
    gate_min_m: float
    gate_max_m: float
    vertical_fov_deg: float
    unreliable_tags: set

    @classmethod
    def load(cls, path="config.yaml"):
        raw = yaml.safe_load(Path(path).read_text())
        ids = sorted(raw["classes"])
        if ids != list(range(len(ids))):
            raise ValueError("config.yaml: class ids must be 0, 1, 2, ... in order")
        gate = raw["range_gate"]
        return cls(
            names=[raw["classes"][i]["name"] for i in ids],
            fsoco_to_class={title: i for i in ids for title in raw["classes"][i]["fsoco"]},
            heights_m=[raw["classes"][i]["height_m"] for i in ids],
            ignore_classes=set(raw["ignore_classes"]),
            keypoints=raw["keypoints"],
            gate_enabled=gate["enabled"],
            gate_min_m=gate["min_m"],
            gate_max_m=gate["max_m"],
            vertical_fov_deg=gate["vertical_fov_deg"],
            unreliable_tags=set(gate["unreliable_tags"]),
        )
