from __future__ import annotations
import os
import time
import traceback
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import segmentation_utils as utils
from find_classes_recid import find_rec_id
from general_utils import LogUtils
from general_utils import _AsyncLastWriter

class ImageSegmentation:
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        model_path: str,
        job_id: str,
        city_id: str,
        main_url: str,
        token: str,
        *,
        xml_data=None,
        depth_main_path=None,
        source_EPSG=None,
        data_type: str | None = None,
        save_img: bool = True,
        save_txt: bool = True,
        save_crops: bool = True,
        pass_postprocess: bool = False,
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder or os.path.join(os.getcwd(), "seg-output")
        self.model_path = model_path
        self.job_id = job_id
        self.city_id = city_id
        self.main_url = main_url
        self.token = token

        self.xml_data = xml_data
        self.depth_main_path = depth_main_path
        self.source_EPSG = source_EPSG
        self.data_type = data_type

        self.save_img = save_img
        self.save_txt = save_txt
        self.save_crops = save_crops

        self.pass_postprocess = pass_postprocess
        self.model = None 

        os.makedirs(self.output_folder, exist_ok=True)

        self.json_log_path = LogUtils.log_path

    def load_model(self) -> None:
        self.model = YOLO(self.model_path)

    def process_images(self, padding: int = 20, mask_alpha: float = 0.3) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_epoch = time.time()
        start_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_epoch))

        try:
            class_dict_df = pd.read_csv("Detection_Classes.csv", encoding="utf-8")
        except FileNotFoundError as exc:
            print("FileNotFoundError: 'Detection_Classes.csv' not found") 

        filenames = sorted(
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        
        image_files: List[str] = [os.path.join(self.input_folder, f) for f in filenames]
        total_images = len(image_files)
        if total_images == 0:
            print("No images found in", self.input_folder)
            return

        start_from = LogUtils.resume_index(self.json_log_path, image_files,
                                   key="Detection / Segmentation")
        if start_from:
            print(f"[RESUME] Skipping first {start_from} images already processed")

        try:
            for img_idx, img_path in enumerate(
                image_files[start_from:], start=start_from + 1
            ):
                fname = os.path.basename(img_path)
                pct_complete = int((img_idx / total_images) * 100)
                print(f"Processing {img_idx}/{total_images}: {fname}")

                results = self.model.predict(source=str(img_path))
                pil_img = Image.open(img_path)

                for res in results:
                    if res.masks is None or res.masks.data is None:
                        continue

                    probs = res.boxes.conf.cpu().numpy().tolist()
                    classes = res.boxes.cls.cpu().numpy().astype(int).tolist()
                    bboxes = res.boxes.xyxy.cpu().numpy().tolist()
                    masks = res.masks.data.cpu().numpy()

                    for i, (mask, cls, prob, bbox) in enumerate(
                        zip(masks, classes, probs, bboxes)
                    ):
                        class_tr = res.names[classes[i]]

                        out_img, b64_img, arr_img = utils.crop_and_annotate(
                            pil_img=pil_img,
                            mask=mask,
                            class_name=class_tr,
                            padding=padding,
                            mask_alpha=mask_alpha,
                        )
                        
                        if out_img is None:
                            raise ValueError("Output image is None. Cannot proceed with processing.")

                        suffix = f"_{class_tr}_{i}"
                        utils.save_results(
                            result=res,
                            image=out_img,
                            output_folder=self.output_folder,
                            save_txt=self.save_txt,
                            save_crops=self.save_crops,
                            save_image=self.save_img,
                            save_conf=True,
                            suffix=suffix,
                        )

                        if self.pass_postprocess:
                            loc_wkt = ""
                            xml_stub = {"Dataset_Name": [""]}
                        else:
                            loc_wkt = utils.main_global_coordinate_calculation(
                                xml_data=self.xml_data,
                                depth_main_path=self.depth_main_path,
                                source_EPSG=self.source_EPSG,
                                data_type=self.data_type,
                                img_id=fname.split(".")[0],
                                boundingbox=bbox
                            )
                            xml_stub = self.xml_data


                LogUtils.update_last_processed(
                    self.json_log_path,
                    key="Detection / Segmentation",
                    idx=img_idx,
                    total=total_images,
                    filename=fname,
                )

                print(f"Finished {fname}\n")
                
        except Exception as exc:
            tb = traceback.format_exc()
            print("Error during processing:\n", tb)
            raise exc
        _AsyncLastWriter.flush_now()