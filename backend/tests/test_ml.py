import os
import sys
import unittest
import torch
import numpy as np
import cv2

# Ensure backend package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ml.dataset import SolarFilamentDataset
from backend.ml.cnn_model import FilamentBasicCNN, compute_cnn_loss
from backend.ml.unet_model import (
    SolarFilamentUNet,
    mask_to_polygons_and_bboxes,
    binary_mask_to_rle,
    create_detection_overlay
)
from backend.ml.losses import DiceBCELoss, compute_segmentation_metrics


class TestSolarMLPipeline(unittest.TestCase):

    def setUp(self):
        self.json_path = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        self.img_dir = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\train_images"

    def test_dataset_loading(self):
        if not os.path.exists(self.json_path):
            self.skipTest("MAGFiLO JSON not found on disk")

        ds = SolarFilamentDataset(
            json_path=self.json_path,
            image_dir=self.img_dir,
            target_size=(256, 256),
            is_train=False,
            max_samples=5
        )
        self.assertGreater(len(ds), 0)
        item = ds[0]

        # Verify shapes & types
        self.assertEqual(item["image"].shape, (1, 256, 256))
        self.assertEqual(item["mask"].shape, (1, 256, 256))
        self.assertIn("has_filament", item)
        self.assertIn("category_id", item)
        self.assertIsInstance(item["has_filament"].item(), float)

    def test_basic_cnn_forward(self):
        model = FilamentBasicCNN(in_channels=1, num_classes=5)
        model.eval()

        dummy_input = torch.randn(2, 1, 256, 256)
        outputs = model(dummy_input)

        self.assertIn("presence_logits", outputs)
        self.assertIn("presence_prob", outputs)
        self.assertIn("category_logits", outputs)
        self.assertIn("category_probs", outputs)

        self.assertEqual(outputs["presence_logits"].shape, (2, 1))
        self.assertEqual(outputs["category_logits"].shape, (2, 5))

        # Check probability bounds
        self.assertTrue((outputs["presence_prob"] >= 0.0).all())
        self.assertTrue((outputs["presence_prob"] <= 1.0).all())

        # Test loss calculation
        targets_presence = torch.tensor([1.0, 0.0])
        targets_category = torch.tensor([1, 0])
        loss, loss_dict = compute_cnn_loss(outputs, targets_presence, targets_category)
        self.assertGreater(loss.item(), 0.0)

    def test_unet_segmenter_forward(self):
        model = SolarFilamentUNet(n_channels=1, n_classes=1)
        model.eval()

        dummy_input = torch.randn(2, 1, 256, 256)
        outputs = model(dummy_input)

        self.assertIn("logits", outputs)
        self.assertIn("probs", outputs)
        self.assertEqual(outputs["logits"].shape, (2, 1, 256, 256))
        self.assertEqual(outputs["probs"].shape, (2, 1, 256, 256))

        # Test loss and metrics
        criterion = DiceBCELoss()
        dummy_targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
        loss, loss_dict = criterion(outputs["logits"], dummy_targets)
        self.assertGreater(loss.item(), 0.0)

        metrics = compute_segmentation_metrics(outputs["probs"], dummy_targets)
        self.assertIn("iou", metrics)
        self.assertIn("dice", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)

    def test_polygon_and_overlay_generation(self):
        # Create a synthetic binary mask with a circle
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 30, 1, -1)

        filaments = mask_to_polygons_and_bboxes(mask, min_area=10, orig_shape=(2048, 2048))
        self.assertGreaterEqual(len(filaments), 1)
        f0 = filaments[0]

        self.assertIn("segmentation", f0)
        self.assertIn("bbox", f0)
        self.assertIn("area", f0)
        self.assertGreater(len(f0["segmentation"][0]), 4)

        # RLE test
        rle = binary_mask_to_rle(mask)
        self.assertIn("size", rle)
        self.assertIn("counts", rle)

        # Overlay generation test
        dummy_img = np.ones((256, 256), dtype=np.uint8) * 128
        overlay = create_detection_overlay(dummy_img, mask, filaments)
        self.assertEqual(overlay.shape, (256, 256, 3))


if __name__ == "__main__":
    unittest.main()
