"""
inference_kernel.py

Kaggle code-competition inference script for the RSNA Knee
Abnormality Detection competition. Runs entirely offline inside
Kaggle's sandboxed notebook environment: no internet access, no
argparse, hardcoded /kaggle/input and /kaggle/working paths.

This is the Kaggle-environment counterpart to the local infer.py --
same model, same preprocessing (mirrors preprocessing/transforms.py
and utils/image_stats.py exactly), same aggregation logic, adapted
to Kaggle's fixed directory layout.

Note: on this Kaggle environment, inputs are nested one level deeper
than the classic convention -- under datasets/<username>/<slug>/ and
competitions/<slug>/ rather than directly under /kaggle/input/<slug>/.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Kaggle environment paths -----------------------------------------
CHECKPOINT_PATH = Path("/kaggle/input/datasets/rahim02/rsna-knee-best-model/best_model.pt")
TEST_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/test_series")
OUTPUT_PATH = Path("/kaggle/working/submission.csv")

FINDING_NAMES = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

BATCH_SIZE = 32
NUM_WORKERS = 2


# --- Model definition (inlined -- mirrors models/baseline_cnn.py) ------


class KneeMRIClassifier(nn.Module):
    """Configurable CNN-based multi-label classifier for knee MRI findings."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_findings: int = 12,
        pretrained: bool = False,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0,
        )
        backbone_out_features = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout_rate)
        self.head = nn.Linear(backbone_out_features, num_findings)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.dropout(features)
        return self.head(features)


# --- Preprocessing (inlined -- mirrors preprocessing/transforms.py and --
# --- utils/image_stats.py exactly) --------------------------------------


def robust_normalize(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Normalize an image to [0, 1] using percentile-based clipping.

    Mirrors utils/image_stats.py's robust_normalize exactly.

    Args:
        image: A 2D NumPy array of raw pixel intensities.
        p_low: Lower percentile bound for clipping (default: 1.0).
        p_high: Upper percentile bound for clipping (default: 99.0).

    Returns:
        A float32 NumPy array with values rescaled to [0, 1].
    """
    lo = np.percentile(image, p_low)
    hi = np.percentile(image, p_high)

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    clipped = np.clip(image, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return normalized.astype(np.float32)


def preprocess_slice(
    raw_pixel_array: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Run the full preprocessing pipeline on one raw DICOM pixel array.

    Mirrors preprocessing/transforms.py's preprocess_slice exactly:
    robust normalize -> resize -> expand to 3 channels.

    Args:
        raw_pixel_array: The raw 2D pixel array from pydicom's
            ds.pixel_array (any integer dtype).
        target_size: Desired (height, width) of the final image.

    Returns:
        A float32 NumPy array of shape (H, W, 3) with values in [0, 1].
    """
    normalized = robust_normalize(raw_pixel_array)
    resized = cv2.resize(
        normalized, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR,
    )
    three_channel = np.stack([resized, resized, resized], axis=-1)
    return three_channel


# --- Dataset (inlined -- mirrors datasets/knee_test_dataset.py) --------


class KneeMRITestDataset(Dataset):
    """Slice-level Dataset for unlabeled inference on Kaggle's test_series."""

    def __init__(self, series_root: Path) -> None:
        self.series_root = series_root

        records = []
        for dcm_path in series_root.rglob("*.dcm"):
            series_id = dcm_path.parent.name
            study_id = dcm_path.parent.parent.name
            records.append({
                "study_id": study_id,
                "series_id": series_id,
                "filename": dcm_path.name,
            })

        if not records:
            raise FileNotFoundError(f"No .dcm files found under {series_root}.")

        self.manifest = pd.DataFrame(records)

        logger.info(
            "KneeMRITestDataset initialized | slices=%d | studies=%d | root=%s",
            len(self.manifest), self.manifest["study_id"].nunique(), series_root,
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        dcm_path = self.series_root / row["study_id"] / row["series_id"] / row["filename"]
        ds = pydicom.dcmread(dcm_path)
        image = preprocess_slice(ds.pixel_array)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return {"image": image_tensor, "study_id": row["study_id"]}


# --- Inference logic (mirrors infer.py) ---------------------------------


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained KneeMRIClassifier, with fallback for older checkpoints.

    Args:
        checkpoint_path: Path to a .pt file saved by train.py.
        device: Device to load the model onto.

    Returns:
        The reconstructed model, in eval() mode, weights loaded.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    backbone_name = checkpoint.get("backbone_name", "resnet18")
    dropout_rate = checkpoint.get("hyperparameters", {}).get("dropout_rate", 0.3)

    model = KneeMRIClassifier(
        backbone_name=backbone_name, pretrained=False, dropout_rate=dropout_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint | backbone=%s | epoch=%s | macro_auc=%s",
        backbone_name, checkpoint.get("epoch"), checkpoint.get("macro_auc"),
    )
    return model


def run_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Run the model over every slice in the test set.

    Args:
        model: A trained model in eval() mode.
        test_loader: DataLoader over a KneeMRITestDataset.
        device: Device to run computation on.

    Returns:
        A DataFrame with one row per slice: study_id plus one
        probability column per finding.
    """
    all_study_ids = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            images = batch["image"].to(device)
            study_ids = batch["study_id"]

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_study_ids.extend(study_ids)

            if batch_idx % 20 == 0:
                logger.info("Inference batch %d", batch_idx)

    probs_array = np.concatenate(all_probs, axis=0)
    slice_df = pd.DataFrame(probs_array, columns=FINDING_NAMES)
    slice_df.insert(0, "study_id", all_study_ids)
    return slice_df


def aggregate_to_study_level(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate slice-level probabilities to one row per study.

    Args:
        slice_df: Output of run_inference().

    Returns:
        A DataFrame with columns StudyInstanceUID plus one column
        per finding.
    """
    study_df = slice_df.groupby("study_id")[FINDING_NAMES].mean().reset_index()
    study_df = study_df.rename(columns={"study_id": "StudyInstanceUID"})
    return study_df


def log_kaggle_input_contents() -> None:
    """Log the contents of /kaggle/input for debugging mount paths.

    Recurses up to three levels deep to fully reveal the actual mount
    structure, since dataset/competition folders are nested under
    datasets/<username>/<slug>/ and competitions/<slug>/ on this
    environment rather than directly under /kaggle/input/<slug>/.
    """
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        logger.info("%s%s/", "  " * depth, root)
        for f in files[:5]:
            logger.info("%s  %s", "  " * depth, f)


def main() -> None:
    """Run the full inference pipeline and write submission.csv."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    log_kaggle_input_contents()

    model = load_model(CHECKPOINT_PATH, device)

    test_dataset = KneeMRITestDataset(series_root=TEST_ROOT)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    slice_df = run_inference(model, test_loader, device)
    study_df = aggregate_to_study_level(slice_df)

    study_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(
        "Submission written | studies=%d | slices=%d | path=%s",
        len(study_df), len(slice_df), OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()