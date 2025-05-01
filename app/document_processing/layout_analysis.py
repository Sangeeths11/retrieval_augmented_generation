import os
import cv2
import torch
from pathlib import Path
from typing import Optional
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10


class DocumentLayoutAnalyzer:
    """
    Analyzes PDF layout using DocLayout-YOLO and saves cropped elements by type.
    Props to: https://github.com/opendatalab/DocLayout-YOLO/tree/main?tab=readme-ov-file
    """

    def __init__(
        self,
        model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench",
        model_filename: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
        output_dir: str = "layout_outputs",
        conf_threshold: float = 0.25,
        image_size: int = 1024,
    ):
        self.device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
        self.model = YOLOv10(model_path)
        self.output_dir = Path(output_dir)
        self.conf = conf_threshold
        self.imgsz = image_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Model loaded on {self.device}")

    def analyze_pdf(self, pdf_path: str):
        """
        Analyze a PDF file and extract layout elements as image crops.

        Args:
            pdf_path: Path to the PDF file.
        """
        print(f"[INFO] Analyzing PDF: {pdf_path}")
        pages = convert_from_path(pdf_path)
        for i, page in enumerate(pages):
            image_path = self.output_dir / f"page_{i}.jpg"
            page.save(image_path)
            self._analyze_image(str(image_path), i)

    def _analyze_image(self, image_path: str, page_number: int):
        """
        Analyze a single image using the YOLOv10 layout model.

        Args:
            image_path: Path to the image file.
            page_number: Page number for output naming.
        """
        print(f"[INFO] Processing page {page_number}...")
        results = self.model.predict(
            image_path,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
        )

        image = cv2.imread(image_path)
        for i, det in enumerate(results[0].boxes):
            xyxy = list(map(int, det.xyxy[0].tolist()))
            cls_id = int(det.cls[0])
            label = self.model.model.names[cls_id]

            crop = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

            label_dir = self.output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)

            out_path = label_dir / f"page{page_number}_det{i}.jpg"
            cv2.imwrite(str(out_path), crop)

        print(f"[INFO] Page {page_number} layout elements saved.")
