import os
import cv2
import numpy as np
import torch
import logging
from PIL import Image
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

RTDETR_DIR = os.path.join(MODELS_DIR, 'rtdetr')
YOLOV8_SEG_PATH = os.path.join(MODELS_DIR, 'comic-text-segmenter-yolov8m.pt')

RTDETR_CLASSES = {0: 'bubble', 1: 'text_bubble', 2: 'text_free'}

MAX_ASPECT_RATIO = 3.0
CHUNK_OVERLAP = 64


class TextBlock:
    def __init__(self, xyxy, cls_name='text', conf=0.0):
        self.xyxy = tuple(int(v) for v in xyxy)
        self.cls_name = cls_name
        self.conf = conf

    def area(self):
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    def __repr__(self):
        return f"TextBlock({self.xyxy}, cls={self.cls_name}, conf={self.conf:.2f})"


def merge_overlapping_blocks(blk_list, iou_thresh=0.3):
    if len(blk_list) <= 1:
        return blk_list

    boxes = np.array([b.xyxy for b in blk_list], dtype=np.float32)
    cls_names = [b.cls_name for b in blk_list]
    merged = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue
        x1, y1, x2, y2 = boxes[i]
        cur_cls = cls_names[i]
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            bx1, by1, bx2, by2 = boxes[j]
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_j = (bx2 - bx1) * (by2 - by1)
                if inter / max(area_j, 1) > iou_thresh:
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    used.add(j)
        merged.append(TextBlock([int(x1), int(y1), int(x2), int(y2)], cls_name=cur_cls))
        used.add(i)

    return merged


class ComicTextDetector:
    _instance = None

    @classmethod
    def get_instance(cls, device='cpu'):
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def __init__(self, device='cpu'):
        self.device = device
        self._rtdetr_model = None
        self._rtdetr_processor = None
        self._yolo_seg = None
        self._load_models()

    def _load_models(self):
        rtdetr_config = os.path.join(RTDETR_DIR, 'config.json')
        rtdetr_weights = os.path.join(RTDETR_DIR, 'model.safetensors')
        if os.path.exists(rtdetr_config) and os.path.exists(rtdetr_weights):
            try:
                from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
                logger.info("Loading RT-DETR v2 text detector...")
                self._rtdetr_processor = RTDetrImageProcessor.from_pretrained(
                    RTDETR_DIR, local_files_only=True
                )
                self._rtdetr_model = RTDetrV2ForObjectDetection.from_pretrained(
                    RTDETR_DIR, local_files_only=True
                )
                self._rtdetr_model.eval()
                self._rtdetr_model.to(self.device)
                logger.info("RT-DETR v2 loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load RT-DETR: {e}", exc_info=True)
                self._rtdetr_model = None

        if os.path.exists(YOLOV8_SEG_PATH):
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8 text segmenter...")
                self._yolo_seg = YOLO(YOLOV8_SEG_PATH)
                logger.info("YOLOv8 segmenter loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 segmenter: {e}", exc_info=True)
                self._yolo_seg = None

        if self._rtdetr_model is None and self._yolo_seg is None:
            raise RuntimeError("No detection models available! Need RT-DETR or YOLOv8 segmenter.")

    def _needs_chunking(self, h, w):
        aspect = max(h, w) / max(min(h, w), 1)
        return aspect > MAX_ASPECT_RATIO

    def _compute_chunks(self, h, w):
        if w >= h:
            chunk_size = h
            total = w
            axis = 'horizontal'
        else:
            chunk_size = w
            total = h
            axis = 'vertical'

        if axis == 'vertical':
            target_chunk = int(w * MAX_ASPECT_RATIO)
        else:
            target_chunk = int(h * MAX_ASPECT_RATIO)

        target_chunk = max(target_chunk, chunk_size)

        chunks = []
        pos = 0
        while pos < total:
            end = min(pos + target_chunk, total)
            if total - end < target_chunk * 0.3 and end < total:
                end = total
            chunks.append((pos, end))
            if end >= total:
                break
            pos = end - CHUNK_OVERLAP

        return chunks, axis

    @torch.no_grad()
    def _detect_rtdetr(self, image_np: np.ndarray, conf_thresh=0.3) -> List[TextBlock]:
        if self._rtdetr_model is None:
            return []

        h, w = image_np.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        inputs = self._rtdetr_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._rtdetr_model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self._rtdetr_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_thresh
        )[0]

        blocks = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            label_val = label.item()
            x1, y1, x2, y2 = box.int().tolist()

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cls_name = RTDETR_CLASSES.get(label_val, 'text')
            if cls_name == 'bubble':
                continue

            if x2 > x1 and y2 > y1:
                blocks.append(TextBlock([x1, y1, x2, y2], cls_name=cls_name, conf=score_val))

        return blocks

    @torch.no_grad()
    def _get_seg_mask(self, image_np: np.ndarray, conf_thresh=0.25) -> np.ndarray:
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self._yolo_seg is not None:
            results = self._yolo_seg.predict(image_np, imgsz=1024, conf=conf_thresh, verbose=False)
            r = results[0]
            if r.masks is not None:
                for mask_tensor in r.masks.data.cpu().numpy():
                    resized = cv2.resize(mask_tensor, (w, h))
                    mask[resized > 0.5] = 255

        return mask

    @torch.no_grad()
    def _detect_single(self, image_np: np.ndarray):
        h, w = image_np.shape[:2]

        blk_list = self._detect_rtdetr(image_np, conf_thresh=0.3)

        if not blk_list and self._yolo_seg is not None:
            results = self._yolo_seg.predict(image_np, imgsz=1024, conf=0.25, verbose=False)
            r = results[0]
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    blk_list.append(TextBlock([x1, y1, x2, y2], cls_name='text_comic', conf=conf))

        mask = self._get_seg_mask(image_np)

        if mask.sum() == 0 and blk_list:
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                pad = 2
                mask[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)] = 255

        return mask, blk_list

    @torch.no_grad()
    def detect(self, image_np: np.ndarray):
        h, w = image_np.shape[:2]

        if not self._needs_chunking(h, w):
            mask, blk_list = self._detect_single(image_np)
            return mask, mask.copy(), blk_list

        chunks, axis = self._compute_chunks(h, w)
        logger.info(f"Chunked detection: {len(chunks)} chunks ({axis}), image {w}x{h}")

        full_mask = np.zeros((h, w), dtype=np.uint8)
        all_blks = []

        for chunk_idx, (start, end) in enumerate(chunks):
            if axis == 'vertical':
                chunk_img = image_np[start:end, :, :]
                y_offset = start
                x_offset = 0
            else:
                chunk_img = image_np[:, start:end, :]
                y_offset = 0
                x_offset = start

            chunk_mask, chunk_blks = self._detect_single(chunk_img)

            if axis == 'vertical':
                full_mask[start:end, :] = np.maximum(full_mask[start:end, :], chunk_mask)
            else:
                full_mask[:, start:end] = np.maximum(full_mask[:, start:end], chunk_mask)

            for blk in chunk_blks:
                bx1, by1, bx2, by2 = blk.xyxy
                all_blks.append(TextBlock([
                    bx1 + x_offset, by1 + y_offset,
                    bx2 + x_offset, by2 + y_offset
                ], cls_name=blk.cls_name, conf=blk.conf))

        all_blks = merge_overlapping_blocks(all_blks)
        min_block_area = max(15 * 15, (h * w) * 0.00002)
        all_blks = [b for b in all_blks if b.area() >= min_block_area]
        logger.info(f"Chunked detection found {len(all_blks)} text blocks total")

        return full_mask, full_mask.copy(), all_blks

    @torch.no_grad()
    def detect_text_mask(self, image_np: np.ndarray, dilate_size: int = 5, dilate_iter: int = 3) -> np.ndarray:
        mask, mask_refined, blk_list = self.detect(image_np)

        final_mask = mask_refined if mask_refined.sum() > 0 else mask

        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

        if dilate_size > 0 and dilate_iter > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            final_mask = cv2.dilate(final_mask, kernel, iterations=dilate_iter)

        return final_mask

    def detect_for_cleaning(self, image_bgr: np.ndarray, dilate_size: int = 5, dilate_iter: int = 2):
        mask, mask_refined, blk_list = self.detect(image_bgr)

        final_mask = mask_refined if mask_refined.sum() > 0 else mask

        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

        if dilate_size > 0 and dilate_iter > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            final_mask = cv2.dilate(final_mask, kernel, iterations=dilate_iter)

        return final_mask, blk_list

    def create_inpaint_mask(self, pil_image: Image.Image, dilate_size: int = 5, dilate_iter: int = 2) -> Image.Image:
        img_np = np.array(pil_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        mask = self.detect_text_mask(img_bgr, dilate_size=dilate_size, dilate_iter=dilate_iter)

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(mask_rgb)
