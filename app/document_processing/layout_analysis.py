import os
import cv2
import torch
from pathlib import Path
from typing import Optional
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from app.core.config import STORAGE_DIR
import json



class DocumentLayoutAnalyzer:
    """
    Analyzes PDF layout using DocLayout-YOLO and saves cropped elements by type.
    Props to: https://github.com/opendatalab/DocLayout-YOLO/tree/main?tab=readme-ov-file
    """

    def __init__(
        self,
        model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench",
        model_filename: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
        conf_threshold: float = 0.25,
        output_dir: Optional[Path] = None,
        image_size: int = 1024,
    ):
        self.device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
        self.model = YOLOv10(model_path)
        self.conf = conf_threshold
        self.imgsz = image_size

        self.base_output_dir = Path(STORAGE_DIR) / "layout_outputs"
        self.base_output_dir.mkdir(parents=True, exist_ok=True)


        print(f"[INFO] Model loaded on {self.device}")

    def analyze_pdf(self, pdf_path: str):
        """
        Analyze a PDF file and extract layout elements as image crops.

        Args:
            pdf_path: Path to the PDF file.
        """

        pdf_path = Path(pdf_path)
        doc_name = pdf_path.stem
        doc_output_dir = self.base_output_dir / doc_name
        doc_output_dir.mkdir(parents=True, exist_ok=True)


        print(f"[INFO] Analyzing PDF: {pdf_path.name}")
        pages = convert_from_path(pdf_path)
        for i, page in enumerate(pages):
            page_dir = doc_output_dir / f"page_{i}"
            page_dir.mkdir(parents=True, exist_ok=True)

            image_path = page_dir / "full.jpg"
            page.save(image_path)
            self._analyze_image(str(image_path), i, page_dir)

    def _analyze_image(self, image_path: str, page_number: int, page_dir: Path):
        """
        Analyze a single page image, save cropped elements + layout.json.
        """
        print(f"[INFO] Processing page {page_number}...")
        results = self.model.predict(
            image_path,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
        )

        image = cv2.imread(image_path)
        layout_data = []
        label_counts = {}

        for i, det in enumerate(results[0].boxes):
            xyxy = list(map(int, det.xyxy[0].tolist()))
            cls_id = int(det.cls[0])
            label = self.model.model.names[cls_id]

            # Track count per label to number crops
            label_counts[label] = label_counts.get(label, 0) + 1
            label_index = label_counts[label] - 1
            crop_filename = f"{label}_{label_index}.jpg"

            crop_path = page_dir / crop_filename
            crop_img = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            cv2.imwrite(str(crop_path), crop_img)

            layout_data.append({
                "id": f"{label}_{label_index}",
                "label": label,
                "page": page_number,
                "bbox": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                },
                "image": crop_filename
            })

        with open(page_dir / "layout.json", "w") as f:
            json.dump(layout_data, f, indent=2)

        print(f"[INFO] Saved {len(layout_data)} layout elements to {page_dir}")

