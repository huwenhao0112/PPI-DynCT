"""Shared runtime helpers for native multi-seed McFarland experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def sample_cell_type(sample) -> str:
    value = getattr(sample, "cell_type", None)
    if value is not None:
        if isinstance(value, (list, tuple)):
            value = value[0]
        if torch.is_tensor(value):
            value = value.flatten()[0].item()
        return str(value)
    return str(sample.cov_drug).split("_", 1)[0]


def sample_drug(sample) -> str:
    value = getattr(sample, "drug", None)
    if value is not None:
        if isinstance(value, (list, tuple)):
            value = value[0]
        if torch.is_tensor(value):
            value = value.flatten()[0].item()
        return str(value)
    cell = sample_cell_type(sample)
    cov_drug = str(sample.cov_drug)
    return cov_drug[len(cell) + 1:] if cov_drug.startswith(cell + "_") else cov_drug.split("_", 1)[-1]


def split_samples(
    samples, ood_cov_drug, testing_cell_types, split_mode="loco",
    testing_drugs=None,
):
    """Split samples while making the LOCO rule explicit.

    ``loco`` removes every sample of a held-out cell line from training, while
    OOD evaluation remains restricted to the configured drug conditions.
    ``lodo`` removes a held-out drug from every cell line.
    ``double_ood`` removes both every sample of a held-out cell line and every
    sample of a held-out drug; evaluation is their observed intersection.
    ``pair`` only removes the configured cell-line/drug pairs.
    """
    ood_set = set(map(str, ood_cov_drug))
    test_cells = set(map(str, testing_cell_types))
    test_drugs = set(map(str, testing_drugs or ()))
    cells_ood = [s for s in samples if str(s.cov_drug) in ood_set]
    if split_mode == "loco":
        cells_train = [s for s in samples if sample_cell_type(s) not in test_cells]
    elif split_mode == "lodo":
        if not test_drugs:
            raise ValueError("split_mode='lodo' requires testing_drugs")
        cells_train = [s for s in samples if sample_drug(s) not in test_drugs]
    elif split_mode == "double_ood":
        if not test_cells or not test_drugs:
            raise ValueError("split_mode='double_ood' requires testing_cell_types and testing_drugs")
        cells_train = [
            s for s in samples
            if sample_cell_type(s) not in test_cells and sample_drug(s) not in test_drugs
        ]
    elif split_mode == "pair":
        cells_train = [s for s in samples if str(s.cov_drug) not in ood_set]
    else:
        raise ValueError(f"Unsupported split_mode={split_mode!r}")
    if not cells_train:
        raise RuntimeError("Training split is empty; check testing_cell_type/split_mode.")
    if not cells_ood:
        raise RuntimeError("OOD split is empty; check testing_cell_type/testing_drugs.")
    overlap = {str(s.cov_drug) for s in cells_train}.intersection(
        str(s.cov_drug) for s in cells_ood
    )
    if overlap:
        raise RuntimeError(f"Train/OOD leakage detected: {sorted(overlap)[:5]}")
    train_cells = {sample_cell_type(s) for s in cells_train}
    train_drugs = {sample_drug(s) for s in cells_train}
    if split_mode in {"loco", "double_ood"} and train_cells.intersection(test_cells):
        raise RuntimeError("Held-out cell type leaked into training")
    if split_mode in {"lodo", "double_ood"} and train_drugs.intersection(test_drugs):
        raise RuntimeError("Held-out drug leaked into training")
    return cells_train, cells_ood


def resolve_run_paths(run_dir, default_results, default_models):
    if not run_dir:
        return None, str(default_results), str(default_models)
    root = Path(run_dir).expanduser().resolve()
    results = root / "results"
    models = root / "models"
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return root, str(results), str(models)


def config_sha256(config_path: str) -> str:
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()


def write_complete_manifest(
    run_dir,
    *,
    native_variant,
    native_script,
    seed,
    config_path,
    split_mode,
    testing_cell_types,
    testing_drugs,
    model_file,
    result_dir,
    node2vec_seed=None,
    extra=None,
):
    if run_dir is None:
        return None
    manifest = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "native_variant": str(native_variant),
        "native_script": str(native_script),
        "seed": int(seed),
        "node2vec_seed": None if node2vec_seed is None else int(node2vec_seed),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "config_sha256": config_sha256(config_path),
        "split_mode": str(split_mode),
        "testing_cell_types": list(map(str, testing_cell_types)),
        "testing_drugs": list(map(str, testing_drugs)),
        "model_file": str(Path(model_file).resolve()),
        "result_dir": str(Path(result_dir).resolve()),
    }
    if extra:
        manifest.update(extra)
    path = Path(run_dir) / "run_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    (Path(run_dir) / "SUCCESS").write_text("complete\n", encoding="utf-8")
    return path

