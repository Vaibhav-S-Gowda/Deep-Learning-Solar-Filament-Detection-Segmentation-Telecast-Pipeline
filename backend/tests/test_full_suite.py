"""
Comprehensive Automated Test Suite for Solar Filament Detection & Segmentation.

Covers:
1. Invalid Images & Edge Cases (Empty files, corrupted bytes, text files, out-of-bounds thresholds, bad base64)
2. API Responses & Contracts (Status codes, response schema validation, error payloads)
3. Model Outputs & Math Invariants (Tensor shapes, probability bounds [0, 1], polygon geometries, loss gradients)
"""

import os
import sys
import io
import json
import base64
import unittest
import numpy as np
import cv2
import torch
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.main import app
from backend.ml.cnn_model import FilamentBasicCNN, compute_cnn_loss
from backend.ml.unet_model import (
    SolarFilamentUNet,
    mask_to_polygons_and_bboxes,
    binary_mask_to_rle
)
from backend.ml.losses import DiceLoss, DiceBCELoss, compute_segmentation_metrics


class TestSolarFilamentComprehensive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Create a valid synthetic 256x256 grayscale image for testing
        synth_img = np.zeros((256, 256), dtype=np.uint8)
        # Draw a synthetic filament ribbon
        cv2.ellipse(synth_img, (128, 128), (80, 20), 45, 0, 360, 255, -1)
        _, buffer = cv2.imencode(".jpg", synth_img)
        cls.valid_image_bytes = buffer.tobytes()
        cls.valid_image_base64 = f"data:image/jpeg;base64,{base64.b64encode(cls.valid_image_bytes).decode('utf-8')}"

    # =========================================================================
    # PART 1: INVALID IMAGES & ERROR HANDLING TESTS
    # =========================================================================

    def test_empty_image_upload_detect(self):
        """Empty 0-byte file upload to /api/v1/detect must return HTTP 400."""
        response = self.client.post(
            "/api/v1/detect",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_empty_image_upload_segment(self):
        """Empty 0-byte file upload to /api/v1/segment must return HTTP 400."""
        response = self.client.post(
            "/api/v1/segment",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_corrupted_image_bytes(self):
        """Corrupted/random non-image bytes must return HTTP 400."""
        corrupted_bytes = b"GARBAGE_HEADER_NOT_A_VALID_JPEG_IMAGE_CONTENT_ABCXYZ123456"
        response = self.client.post(
            "/api/v1/segment",
            files={"file": ("corrupt.jpg", corrupted_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid", response.json()["detail"].lower())

    def test_text_file_as_image(self):
        """Uploading a plain text file pretending to be an image must return HTTP 400."""
        text_content = b"This is a plain text file, not a telescope observation."
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("fake_image.txt", text_content, "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid image", response.json()["detail"].lower())

    def test_invalid_threshold_negative(self):
        """Negative threshold must trigger HTTP 400."""
        response = self.client.post(
            "/api/v1/segment",
            files={"file": ("solar.jpg", self.valid_image_bytes, "image/jpeg")},
            data={"threshold": -0.25}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("threshold", response.json()["detail"].lower())

    def test_invalid_threshold_exceeds_one(self):
        """Threshold >= 1.0 must trigger HTTP 400."""
        response = self.client.post(
            "/api/v1/segment",
            files={"file": ("solar.jpg", self.valid_image_bytes, "image/jpeg")},
            data={"threshold": 1.5}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("threshold", response.json()["detail"].lower())

    def test_telecast_invalid_base64_string(self):
        """Malformed base64 payload to telecast stream must return HTTP 400."""
        response = self.client.post(
            "/api/v1/telecast/process-frame",
            json={"frame_base64": "NOT_A_VALID_BASE64_STRING_%%%!!!"}
        )
        self.assertEqual(response.status_code, 400)

    # =========================================================================
    # PART 2: API RESPONSES & CONTRACT TESTS
    # =========================================================================

    def test_health_endpoint(self):
        """Verifies service health check response."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("device", data)
        self.assertIn("stage1_cnn_ready", data)
        self.assertIn("stage2_unet_ready", data)

    def test_root_endpoint_metadata(self):
        """Verifies root endpoint documentation index."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

    def test_valid_detect_api(self):
        """Stage 1 /api/v1/detect response schema validation."""
        response = self.client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", self.valid_image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["has_filament"], bool)
        self.assertTrue(0.0 <= data["filament_probability"] <= 1.0)
        self.assertIn("dominant_category", data)
        self.assertIsInstance(data["category_scores"], list)
        self.assertGreater(data["inference_time_ms"], 0.0)

    def test_valid_segment_api(self):
        """Stage 2 /api/v1/segment response schema validation."""
        response = self.client.post(
            "/api/v1/segment",
            files={"file": ("test.jpg", self.valid_image_bytes, "image/jpeg")},
            data={"threshold": 0.4}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["num_filaments_found"], int)
        self.assertGreaterEqual(data["total_filament_area"], 0.0)
        self.assertGreaterEqual(data["coverage_percentage"], 0.0)
        self.assertIsInstance(data["has_high_solar_activity"], bool)
        self.assertIsInstance(data["filaments"], list)
        self.assertIsNotNone(data["rle_mask"])
        self.assertIsNotNone(data["binary_mask_base64"])
        self.assertGreater(data["inference_time_ms"], 0.0)

    def test_valid_analyze_api(self):
        """Full Pipeline /api/v1/analyze response schema validation."""
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", self.valid_image_bytes, "image/jpeg")},
            data={"threshold": 0.4}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("stage1_cnn", data)
        self.assertIn("stage2_segmentation", data)
        self.assertIn("overlay_image_base64", data)
        self.assertTrue(data["overlay_image_base64"].startswith("data:image/jpeg;base64,"))
        self.assertGreater(data["total_processing_time_ms"], 0.0)

    def test_valid_telecast_process_frame(self):
        """Live Telecast stream frame processing validation."""
        response = self.client.post(
            "/api/v1/telecast/process-frame",
            json={
                "frame_base64": self.valid_image_base64,
                "detection_threshold": 0.4,
                "return_overlay": True
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["filaments_detected"], int)
        self.assertIsInstance(data["total_filament_area"], float)
        self.assertIsInstance(data["has_high_solar_activity"], bool)
        self.assertGreater(data["processing_time_ms"], 0.0)

    # =========================================================================
    # PART 3: MODEL OUTPUTS & MATHEMATICAL INVARIANTS
    # =========================================================================

    def test_cnn_output_shapes_and_probabilities(self):
        """Verifies FilamentBasicCNN forward tensor shapes and probability bounds."""
        model = FilamentBasicCNN(in_channels=1, num_classes=5)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 1, 256, 256)
        with torch.no_grad():
            outputs = model(x)

        # Check tensor shapes
        self.assertEqual(outputs["presence_logits"].shape, (batch_size, 1))
        self.assertEqual(outputs["presence_prob"].shape, (batch_size, 1))
        self.assertEqual(outputs["category_logits"].shape, (batch_size, 5))
        self.assertEqual(outputs["category_probs"].shape, (batch_size, 5))
        self.assertEqual(outputs["features"].shape, (batch_size, 256))

        # Check probability bounds
        probs = outputs["presence_prob"]
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        # Softmax probabilities must sum to 1.0 across classes
        cat_sums = outputs["category_probs"].sum(dim=-1)
        for s in cat_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_unet_output_shapes_and_bounds(self):
        """Verifies SolarFilamentUNet forward shapes and sigmoid probability range."""
        model = SolarFilamentUNet(n_channels=1, n_classes=1, base_filters=16)
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, 1, 128, 128)
        with torch.no_grad():
            outputs = model(x)

        self.assertEqual(outputs["logits"].shape, (batch_size, 1, 128, 128))
        self.assertEqual(outputs["probs"].shape, (batch_size, 1, 128, 128))

        probs = outputs["probs"]
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_polygon_geometry_extraction(self):
        """Verifies contour extraction produces valid COCO polygon coordinates and bboxes."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Create two distinct filament-like connected components
        cv2.rectangle(mask, (30, 30), (80, 60), 1, -1)
        cv2.circle(mask, (180, 180), 25, 1, -1)

        filaments = mask_to_polygons_and_bboxes(mask, min_area=10, orig_shape=(1024, 1024))
        self.assertEqual(len(filaments), 2)

        for f in filaments:
            self.assertIn("id", f)
            self.assertIn("segmentation", f)
            self.assertIn("bbox", f)
            self.assertIn("area", f)
            self.assertGreater(f["area"], 0.0)

            # Check bbox structure [x, y, w, h]
            bx, by, bw, bh = f["bbox"]
            self.assertGreater(bw, 0.0)
            self.assertGreater(bh, 0.0)

            # Check polygon coordinate pairs (must be even length >= 6 for a valid polygon)
            poly = f["segmentation"][0]
            self.assertEqual(len(poly) % 2, 0)
            self.assertGreaterEqual(len(poly), 6)

    def test_dice_bce_loss_gradients(self):
        """Verifies DiceBCELoss computes non-negative loss and valid backprop gradients."""
        criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
        model = SolarFilamentUNet(n_channels=1, n_classes=1, base_filters=16)
        model.train()

        x = torch.randn(2, 1, 64, 64)
        targets = (torch.rand(2, 1, 64, 64) > 0.7).float()

        outputs = model(x)
        loss, loss_dict = criterion(outputs["logits"], targets)

        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertIn("loss_bce", loss_dict)
        self.assertIn("loss_dice", loss_dict)

        # Backpropagation test
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())


if __name__ == "__main__":
    unittest.main()
