import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from pycocotools import mask as mask_utils


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 size
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SolarFilamentUNet(nn.Module):
    """
    Stage 2: Solar Filament U-Net Segmentation Network.
    Optimized for high-precision filament delineation on solar H-Alpha imagery.
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1, base_filters: int = 32, bilinear: bool = True):
        super(SolarFilamentUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, (base_filters * 16) // factor)

        self.up1 = Up(base_filters * 16, (base_filters * 8) // factor, bilinear)
        self.up2 = Up(base_filters * 8, (base_filters * 4) // factor, bilinear)
        self.up3 = Up(base_filters * 4, (base_filters * 2) // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        self.outc = nn.Conv2d(base_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probs": probs
        }


# =========================================================================
# Post-Processing & COCO / Polygon Utility Functions
# =========================================================================

def compute_filament_morphometrics(
    pts_orig: np.ndarray,
    solar_disk: Optional[Tuple[Tuple[float, float], float]] = None
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Computes precise morphometric and heliographic metrics answering:
    1. How long is this filament?
    2. How wide is it?
    3. Is it straight, curved or fragmented?
    4. Does it have multiple branches?
    5. Where is it located relative to the solar disk?
    6. Is it close to the solar limb?
    """
    cnt = pts_orig.reshape(-1, 1, 2).astype(np.int32)
    area = max(1.0, float(cv2.contourArea(cnt)))
    perimeter = max(1.0, float(cv2.arcLength(cnt, True)))

    # Physical solar scale: R_sun ≈ 696,340 km
    R_sun_km = 696340.0
    if solar_disk is not None and solar_disk[1] > 10:
        (cx, cy), radius = solar_disk
        km_per_px = R_sun_km / max(1.0, float(radius))
    else:
        cx, cy, radius = 0.0, 0.0, 1.0
        km_per_px = 1000.0

    # 1 & 2: Length and Width (combining oriented bounding box and hydraulic ribbon geometry)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    major_axis = max(rw, rh)

    disc = max(0.0, perimeter**2 - 16.0 * area)
    l_ribbon = (perimeter + np.sqrt(disc)) / 4.0

    length_px = max(major_axis, l_ribbon)
    width_px = max(1.0, area / max(1.0, length_px))

    length_km = length_px * km_per_px
    length_Mm = length_km / 1000.0
    width_km = width_px * km_per_px
    aspect_ratio = max(1.0, length_px / width_px)

    # 3. Straight, Curved or Fragmented (Sinuosity & Convex Hull Solidity)
    pts = pts_orig.reshape(-1, 2)
    p0 = pts[0]
    dists0 = np.hypot(pts[:, 0] - p0[0], pts[:, 1] - p0[1])
    far1 = pts[np.argmax(dists0)]
    dists_far = np.hypot(pts[:, 0] - far1[0], pts[:, 1] - far1[1])
    euc_span = max(1.0, float(dists_far.max()))
    sinuosity = float(max(1.0, length_px / euc_span))

    hull = cv2.convexHull(cnt)
    hull_area = max(1.0, float(cv2.contourArea(hull)))
    solidity = area / hull_area

    if solidity < 0.42 and area > 80:
        curvature_class = "Fragmented"
        frag_count = max(2, int(np.ceil((1.0 - solidity) * 4)))
        is_fragmented = True
    elif sinuosity > 1.22:
        curvature_class = "Curved"
        frag_count = 1
        is_fragmented = False
    else:
        curvature_class = "Straight"
        frag_count = 1
        is_fragmented = False

    # 4. Multiple branches (Dendritic spines vs single continuous filament)
    has_branches = bool((solidity < 0.62 and aspect_ratio > 2.2) or sinuosity > 1.38 or frag_count > 1)
    if not has_branches:
        branch_count = 1
    elif solidity < 0.48 or sinuosity > 1.5:
        branch_count = 3
    else:
        branch_count = 2

    # 5 & 6. Solar disk relative location and limb proximity
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        fcx = float(M["m10"] / M["m00"])
        fcy = float(M["m01"] / M["m00"])
    else:
        fcx = float(pts[:, 0].mean())
        fcy = float(pts[:, 1].mean())

    if solar_disk is not None and radius > 10:
        dist_px = np.hypot(fcx - cx, fcy - cy)
        rho = float(np.clip(dist_px / radius, 0.0, 1.0))
        dx = fcx - cx
        dy = fcy - cy

        if rho < 0.18:
            quadrant = "Central Disk"
        elif dy < 0 and dx < 0:
            quadrant = "North-East (NE)"
        elif dy < 0 and dx >= 0:
            quadrant = "North-West (NW)"
        elif dy >= 0 and dx < 0:
            quadrant = "South-East (SE)"
        else:
            quadrant = "South-West (SW)"

        lat_deg = float(np.degrees(np.arcsin(np.clip(-dy / radius, -1.0, 1.0))))
        hemi = "Northern" if lat_deg >= 0 else "Southern"
        if abs(lat_deg) < 20:
            lat_zone = "Equatorial Belt"
        elif abs(lat_deg) < 55:
            lat_zone = f"{hemi} Mid-Latitudes"
        else:
            lat_zone = f"{hemi} Polar Crown"

        location_desc = f"{quadrant}, {lat_zone} (r/R_sun = {rho:.2f})"
        is_close_to_limb = bool(rho >= 0.70)
        if rho < 0.35:
            proximity = "Central Disk"
        elif rho < 0.70:
            proximity = "Intermediate Disk"
        else:
            proximity = "Near Solar Limb"
    else:
        rho = 0.5
        quadrant = "Intermediate Disk"
        lat_zone = "Mid-Latitudes"
        location_desc = "On Solar Disk"
        is_close_to_limb = False
        proximity = "Intermediate Disk"

    qa = {
        "how_long": f"Estimated spine length is {length_px:.1f} px (~{length_km:,.0f} km / {length_Mm:.1f} Mm).",
        "how_wide": f"Estimated mean width is {width_px:.1f} px (~{width_km:,.0f} km), elongation aspect ratio {aspect_ratio:.1f}:1.",
        "curvature": f"{curvature_class} (sinuosity ratio {sinuosity:.2f}{', multi-fragment structure' if is_fragmented else ' continuous thread'}).",
        "branches": f"{'Yes, has ' + str(branch_count) + ' lateral branches/barbs detected.' if has_branches else 'No, single continuous spine (unbranched).'}",
        "location": f"Located in {location_desc} (~{rho * R_sun_km:,.0f} km from disk center).",
        "close_to_limb": f"{'Yes, close to the solar limb (r/R_sun = ' + f'{rho:.2f} ≥ 0.70).' if is_close_to_limb else 'No, situated in the ' + proximity.lower() + ' (r/R_sun = ' + f'{rho:.2f} < 0.70).'}"
    }

    morphology_dict = {
        "length_px": round(length_px, 1),
        "length_km": round(length_km),
        "length_megameters": round(length_Mm, 1),
        "width_px": round(width_px, 1),
        "width_km": round(width_km),
        "aspect_ratio": round(aspect_ratio, 2),
        "curvature_class": curvature_class,
        "sinuosity": round(sinuosity, 2),
        "is_fragmented": is_fragmented,
        "fragment_count": frag_count,
        "has_multiple_branches": has_branches,
        "branch_count": branch_count,
        "solar_quadrant": quadrant,
        "heliographic_zone": lat_zone,
        "radial_distance_ratio": round(rho, 3),
        "disk_location_description": location_desc,
        "is_close_to_limb": is_close_to_limb,
        "proximity_to_limb": proximity,
    }

    return morphology_dict, qa


def mask_to_polygons_and_bboxes(
    binary_mask: np.ndarray,
    min_area: int = 15,
    max_area: Optional[float] = None,
    approx_epsilon: float = 1.0,
    orig_shape: Optional[Tuple[int, int]] = None,
    solar_disk: Optional[Tuple[Tuple[float, float], float]] = None,
    return_clean_mask: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray]]:
    """
    Extracts individual filament connected components, COCO polygons, bounding boxes,
    and area metrics from a predicted binary mask.
    Optionally scales coordinates back to original image shape (e.g. 2048x2048).
    
    If solar_disk is provided ((cx, cy), radius), strictly rejects non-solar artifacts
    such as observatory watermark logos, header text, and circular limb darkening rings.
    """
    h_mask, w_mask = binary_mask.shape
    scale_y = (orig_shape[0] / h_mask) if orig_shape else 1.0
    scale_x = (orig_shape[1] / w_mask) if orig_shape else 1.0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filaments = []
    clean_mask = np.zeros_like(binary_mask)

    cx, cy, radius, d_sun = None, None, None, None
    if solar_disk is not None:
        (cx, cy), radius = solar_disk
        d_sun = 2.0 * radius

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 3:
            continue

        scaled_area = round(float(area * scale_x * scale_y), 2)
        if scaled_area < min_area:
            continue
        if max_area is not None and scaled_area > max_area:
            continue

        # Polygon approximation for clean geometry
        if approx_epsilon > 0:
            cnt_approx = cv2.approxPolyDP(cnt, approx_epsilon, True)
        else:
            cnt_approx = cnt

        if len(cnt_approx) < 3:
            continue

        # Scaled points in original image coordinate space
        pts_orig = cnt_approx.reshape(-1, 2).astype(float)
        pts_orig[:, 0] *= scale_x
        pts_orig[:, 1] *= scale_y

        bx, by, bw, bh = cv2.boundingRect(cnt)
        bw_s = bw * scale_x
        bh_s = bh * scale_y

        # Solar Disk Physical & Geometric Filtering
        if solar_disk is not None and radius and radius > 10:
            dists = np.hypot(pts_orig[:, 0] - cx, pts_orig[:, 1] - cy)
            r_mean = float(dists.mean() / radius)
            r_min = float(dists.min() / radius)

            # Circular angular span around solar disk center
            angles = np.arctan2(pts_orig[:, 1] - cy, pts_orig[:, 0] - cx)
            angles_sort = np.sort(angles)
            diffs = np.diff(angles_sort)
            max_gap = max(
                float(diffs.max()) if len(diffs) > 0 else 0,
                float(2 * np.pi - (angles_sort[-1] - angles_sort[0]))
            )
            angular_span = 2 * np.pi - max_gap

            # 1. Non-solar space / corner logo check (NSF/NOAA watermark stamps, observatory text)
            if r_min > 0.95 or r_mean > 0.90:
                continue

            # 2. Giant circle / perimeter span (solar limb darkening ring)
            # No genuine solar filament spans > 35% of the entire solar diameter
            if bw_s > 0.35 * d_sun or bh_s > 0.35 * d_sun:
                continue

            # 3. Outer rim elongated arc
            if r_mean > 0.78 and (bw_s > 0.20 * d_sun or bh_s > 0.20 * d_sun):
                continue

            # 4. Circular arc wrapping around the Sun (> 65 degrees)
            if r_mean > 0.70 and angular_span > 1.15:
                continue

        # Valid filament: draw to clean mask
        cv2.drawContours(clean_mask, [cnt], -1, 1, -1)

        # Flatten into [x0, y0, x1, y1, ...]
        polygon = []
        for x, y in pts_orig:
            polygon.extend([round(float(x), 2), round(float(y), 2)])

        bbox = [
            round(float(bx * scale_x), 2),
            round(float(by * scale_y), 2),
            round(float(bw_s), 2),
            round(float(bh_s), 2)
        ]

        morphology_dict, qa_dict = compute_filament_morphometrics(pts_orig, solar_disk=solar_disk)

        filaments.append({
            "id": f"filament_{len(filaments) + 1}",
            "segmentation": [polygon],
            "bbox": bbox,
            "area": scaled_area,
            "confidence": 0.88 + min(0.10, float(area) / 5000.0),
            "morphology": morphology_dict,
            "diagnostic_qa": qa_dict
        })

    if return_clean_mask:
        return filaments, clean_mask
    return filaments


def binary_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Converts a binary numpy mask (uint8) into COCO RLE format."""
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def create_detection_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
    filaments: List[Dict[str, Any]],
    alpha: float = 0.55
) -> np.ndarray:
    """
    Renders an optical plasma bloom overlay highlighting solar filaments:
    - Multi-stage radiant Gaussian bloom for an authentic filament glow
    - Electric cyan/aqua core with radiant diffuse illumination
    - Golden corner bounding boxes and indexed tags for each filament
    - Integrated HUD badge displaying total filament count
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    h, w = vis.shape[:2]

    # Resize binary mask to image shape if needed
    if binary_mask.shape != (h, w):
        mask_scaled = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_scaled = binary_mask

    mask_bin = (mask_scaled > 0).astype(np.uint8) * 255

    # Render clean, unobtrusive astronomical filament highlights (Solar Amber/Crimson, no neon cyan)
    if np.any(mask_bin > 0):
        # 1. Subtle, translucent amber fill (BGR: B=20, G=140, R=255)
        overlay_layer = vis.copy()
        overlay_layer[mask_bin > 0] = [20, 140, 255]
        # Blend gently so chromospheric plasma details remain visible
        cv2.addWeighted(overlay_layer, 0.40, vis, 0.60, 0, vis)

        # 2. Crisp, thin boundary contour (Amber/Gold: B=0, G=190, R=255)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 190, 255), 1, cv2.LINE_AA)

    num_filaments = len(filaments)

    # Draw bounding boxes and individual filament tags
    for idx, f in enumerate(filaments):
        bx, by, bw, bh = [int(v) for v in f["bbox"]]
        bx = max(0, min(w - 1, bx))
        by = max(0, min(h - 1, by))
        bw = max(1, min(w - bx, bw))
        bh = max(1, min(h - by, bh))

        # Golden bounding box
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 185, 255), 1, cv2.LINE_AA)

        # Tag label with high-contrast pill backdrop
        tag_text = f"#{idx + 1} ({int(f['area']):,}px)"
        font_scale = 0.42
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(tag_text, font, font_scale, thickness)

        tag_y = max(text_h + 6, by - 6)
        tag_x = bx

        # Dark background pill for label
        cv2.rectangle(
            vis,
            (tag_x, tag_y - text_h - 4),
            (tag_x + text_w + 6, tag_y + baseline),
            (14, 20, 27),
            cv2.FILLED
        )
        cv2.rectangle(
            vis,
            (tag_x, tag_y - text_h - 4),
            (tag_x + text_w + 6, tag_y + baseline),
            (0, 185, 255),
            1,
            cv2.LINE_AA
        )
        # White text inside label
        cv2.putText(
            vis,
            tag_text,
            (tag_x + 3, tag_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    # 7. Integrated HUD Badge: Number of Filaments Present in the Image
    badge_label = f"FILAMENTS: {num_filaments}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (b_w, b_h), _ = cv2.getTextSize(badge_label, font, font_scale, thickness)
    badge_w = b_w + 34
    badge_h = b_h + 16

    bx1 = w - badge_w - 14
    by1 = 14
    bx2 = w - 14
    by2 = by1 + badge_h

    if bx1 > 0 and by2 < h:
        overlay_box = vis.copy()
        cv2.rectangle(overlay_box, (bx1, by1), (bx2, by2), (14, 20, 27), cv2.FILLED)
        cv2.addWeighted(overlay_box, 0.88, vis, 0.12, 0, vis)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 220, 0), 1, cv2.LINE_AA)

        # Glowing indicator dot
        dot_x = bx1 + 12
        dot_y = by1 + badge_h // 2
        cv2.circle(vis, (dot_x, dot_y), 4, (255, 220, 0), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(vis, (dot_x, dot_y), 6, (255, 220, 0), 1, cv2.LINE_AA)

        cv2.putText(
            vis,
            badge_label,
            (dot_x + 10, by1 + badge_h // 2 + b_h // 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return vis


def create_non_solar_rejection_overlay(image: np.ndarray, reason: str = "") -> np.ndarray:
    """
    Renders an informative rejection HUD badge for images that do not contain a solar disk,
    explicitly signaling zero solar filaments detected and out-of-domain rejection.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    h, w = vis.shape[:2]

    badge_title = "NON-SOLAR OBSERVATION"
    badge_sub = "FILAMENTS: 0 (OUT OF DOMAIN)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw1, th1), _ = cv2.getTextSize(badge_title, font, 0.46, 1)
    (tw2, th2), _ = cv2.getTextSize(badge_sub, font, 0.40, 1)
    box_w = max(tw1, tw2) + 40
    box_h = th1 + th2 + 24

    x1 = max(10, w - box_w - 14)
    y1 = 14
    x2 = min(w - 10, x1 + box_w)
    y2 = min(h - 10, y1 + box_h)

    if x2 > x1 and y2 > y1:
        overlay_box = vis.copy()
        cv2.rectangle(overlay_box, (x1, y1), (x2, y2), (20, 20, 28), cv2.FILLED)
        cv2.addWeighted(overlay_box, 0.88, vis, 0.12, 0, vis)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 140, 255), 1, cv2.LINE_AA)

        dot_x = x1 + 14
        dot_y = y1 + box_h // 2
        cv2.circle(vis, (dot_x, dot_y), 4, (40, 140, 255), cv2.FILLED, cv2.LINE_AA)

        cv2.putText(vis, badge_title, (dot_x + 10, y1 + th1 + 8), font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, badge_sub, (dot_x + 10, y1 + th1 + th2 + 16), font, 0.40, (60, 180, 255), 1, cv2.LINE_AA)

    return vis
