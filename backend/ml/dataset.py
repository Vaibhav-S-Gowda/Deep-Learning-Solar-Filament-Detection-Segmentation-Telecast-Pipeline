import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class SolarFilamentDataset(Dataset):
    """
    PyTorch Dataset for MAGFiLO Solar Filament Dataset.
    Loads GONG H-Alpha 2048x2048 grayscale images, resizes to target_size,
    and produces:
      - normalized image tensor (1, H, W)
      - binary segmentation mask tensor (1, H, W)
      - presence label (1.0 if any filament is present, else 0.0)
      - dominant category id (0: None, 1: Left, 2: Right, 3: Unidentifiable, 4: Ambiguous)
      - metadata (file_name, original_size, num_filaments)
    """

    CATEGORIES = {
        0: "None",
        1: "Left",
        2: "Right",
        3: "Unidentifiable",
        4: "Ambiguous"
    }

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        target_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
        max_samples: Optional[int] = None
    ):
        self.json_path = json_path
        self.image_dir = image_dir
        self.target_size = target_size
        self.is_train = is_train

        # Load COCO annotations
        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        self.info = coco_data.get("info", {})
        self.licenses = coco_data.get("licenses", [])
        self.categories_raw = coco_data.get("categories", [])

        # Group annotations by image_id
        self.img_to_anns: Dict[str, List[Dict[str, Any]]] = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Filter images that actually exist on disk
        self.images: List[Dict[str, Any]] = []
        for img_info in coco_data.get("images", []):
            fn = img_info["file_name"]
            full_path = os.path.join(image_dir, fn)
            if os.path.exists(full_path):
                self.images.append(img_info)

        if max_samples and max_samples < len(self.images):
            self.images = self.images[:max_samples]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        orig_h = img_info.get("height", 2048)
        orig_w = img_info.get("width", 2048)

        img_path = os.path.join(self.image_dir, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to zero image
            img = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Build full-resolution ground truth binary mask
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        anns = self.img_to_anns.get(img_id, [])

        cat_counts: Dict[int, int] = {}
        bboxes = []
        for ann in anns:
            cat_id = ann.get("category_id", 0)
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
            if "bbox" in ann:
                bboxes.append(ann["bbox"])

            # Rasterize polygons
            seg_list = ann.get("segmentation", [])
            for poly in seg_list:
                if len(poly) >= 6:
                    pts = np.array(poly, dtype=np.float32).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)

        # Resize image and mask to target_size
        th, tw = self.target_size
        img_resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Optional data augmentation for training
        if self.is_train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_resized = np.fliplr(img_resized)
                mask_resized = np.fliplr(mask_resized)
            # Random vertical flip
            if np.random.rand() > 0.5:
                img_resized = np.flipud(img_resized)
                mask_resized = np.flipud(mask_resized)

        # Normalize image to [0.0, 1.0]
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask_resized.astype(np.float32)).unsqueeze(0)        # (1, H, W)

        has_filament = 1.0 if len(anns) > 0 and np.sum(mask_resized) > 0 else 0.0

        # Dominant category (1..4, or 0 if none)
        if cat_counts:
            dominant_category = max(cat_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant_category = 0

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "has_filament": torch.tensor(has_filament, dtype=torch.float32),
            "category_id": torch.tensor(dominant_category, dtype=torch.long),
            "file_name": file_name,
            "image_id": img_id,
            "original_size": (orig_h, orig_w),
            "num_filaments": len(anns)
        }


def get_dataloaders(
    json_path: str,
    image_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, SolarFilamentDataset]:
    """
    Split dataset into train and validation DataLoaders.
    """
    full_dataset = SolarFilamentDataset(
        json_path=json_path,
        image_dir=image_dir,
        target_size=target_size,
        is_train=False,
        max_samples=max_samples
    )

    total_len = len(full_dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_len), [train_len, val_len], generator=generator
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, full_dataset
