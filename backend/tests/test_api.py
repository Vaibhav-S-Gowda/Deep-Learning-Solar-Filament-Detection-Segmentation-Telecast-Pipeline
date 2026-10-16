import os
import sys
import unittest
import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient

# Ensure backend package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.main import app


class TestSolarBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Create a sample synthetic solar disk image
        synth_img = np.ones((256, 256), dtype=np.uint8) * 40
        # Draw solar disc
        cv2.circle(synth_img, (128, 128), 100, 180, -1)
        # Draw dark filament structure
        cv2.ellipse(synth_img, (128, 110), (40, 6), 25, 0, 360, 30, -1)

        success, buf = cv2.imencode(".jpg", synth_img)
        cls.test_image_bytes = buf.tobytes()
        cls.test_image_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("device", data)

    def test_dataset_stats(self):
        response = self.client.get("/api/v1/dataset/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dataset_name", data)
        self.assertIn("total_train_images", data)
        self.assertIn("total_annotations", data)

    def test_dataset_sample(self):
        response = self.client.get("/api/v1/dataset/sample")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("overlay_base64", data)
        self.assertIn("file_name", data)

    def test_telecast_status(self):
        response = self.client.get("/api/v1/telecast/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")

    def test_telecast_sample_frame(self):
        response = self.client.get("/api/v1/telecast/sample-frame")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("frame_base64", data)

    def test_detect_endpoint(self):
        files = {"file": ("solar_test.jpg", self.test_image_bytes, "image/jpeg")}
        response = self.client.post("/api/v1/detect", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("has_filament", data)
        self.assertIn("filament_probability", data)
        self.assertIn("dominant_category", data)

    def test_segment_endpoint(self):
        files = {"file": ("solar_test.jpg", self.test_image_bytes, "image/jpeg")}
        data_form = {"threshold": 0.3}
        response = self.client.post("/api/v1/segment", files=files, data=data_form)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("num_filaments_found", data)
        self.assertIn("filaments", data)

    def test_analyze_endpoint(self):
        files = {"file": ("solar_test.jpg", self.test_image_bytes, "image/jpeg")}
        data_form = {"threshold": 0.3}
        response = self.client.post("/api/v1/analyze", files=files, data=data_form)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("stage1_cnn", data)
        self.assertIn("stage2_segmentation", data)
        self.assertIn("overlay_image_base64", data)

    def test_telecast_process_frame(self):
        payload = {
            "frame_base64": self.test_image_b64,
            "return_overlay": True,
            "detection_threshold": 0.3
        }
        response = self.client.post("/api/v1/telecast/process-frame", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("filaments_detected", data)
        self.assertIn("overlay_frame_base64", data)

    def test_training_status(self):
        response = self.client.get("/api/v1/train/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("is_training", data)
        self.assertIn("status_message", data)


if __name__ == "__main__":
    unittest.main()
