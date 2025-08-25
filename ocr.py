import os, logging
import numpy as np
from paddleocr import PaddleOCR

class SpeedOCRProcessor:
    def __init__(self, ocr_engine=None):
        self.ocr = ocr_engine if ocr_engine else PaddleOCR(
            use_angle_cls=True, 
            lang='en',
            det_db_thresh=0.2, 
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            text_rec_score_thresh = 0.4
        )
        
    def process_image_array(self, image_data):
        if not isinstance(image_data, np.ndarray):
            return None
        result = self.ocr.ocr(image_data, cls=True)
        if not result or result == [None]:
            return None
        ocr_results = []
        for res in result:
            for line in res:
                detected_text = line[1][0]
                ocr_results.append(detected_text)
        return ocr_results

class HeightOCRProcessor:
    def __init__(self, ocr_engine=None, debug=False):
        self.ocr = ocr_engine if ocr_engine else PaddleOCR(
            lang='en',
            use_angle_cls=True,
            det_db_thresh=0.05,
            det_db_box_thresh=0.05,
            det_db_unclip_ratio=3.0,
            text_rec_score_thresh = 0.4
        )
        self.debug = debug

    def process_image_array(self, image_data):
        raw_result = self.ocr.ocr(image_data, cls=True)
        if not raw_result or raw_result == [None]:
            if self.debug:
                print("No text detected in image data")
            return None, None
        ocr_texts = [word[1][0] for line in raw_result for word in line]
        if self.debug:
            print(f"Processed OCR results: {ocr_texts}")
        return ocr_texts, raw_result

