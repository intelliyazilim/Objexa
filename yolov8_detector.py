import os
import time
import ast
import traceback
from unittest import case
import cv2
from ultralytics import YOLO
from find_classes_recid import find_rec_id
import pandas as pd
import math
import pyproj
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import glob
import uuid
from post_process import check_detection_fp_inthreshold
from pathlib import Path
import re
from ocr import HeightOCRProcessor, SpeedOCRProcessor
from ocr_preprocess import HeightImageResizer, SpeedImageResizer
from ocr_utils import OCRUtils
from general_utils import LogUtils, _AsyncLastWriter, DescripitionBuilder

class YoloV8Detector:

    def __init__(self, job_id, datatype, main_directory, input_directory, model_weights, model_parameters,xml_data,depth_main_path, source_EPSG,city, main_url,token, *, pass_postprocess=False):
        self.pass_postprocess = pass_postprocess

        self.datatype = datatype
        self.main_directory = main_directory
        self.input_directories=input_directory
        self.job_id = job_id
        self.out_directory_directory=os.path.join(Path(__file__).parent,'tmp')
        self.output_job_directory = os.path.join(self.out_directory_directory, self.job_id)
        self.model_parameters = model_parameters
        self.model_weights = model_weights
        self.model = YOLO(model_weights)
        
        self.ocr_height_model = HeightOCRProcessor()
        self.ocr_speed_model = SpeedOCRProcessor()

        self.xml_data= xml_data
        self.source_EPSG=source_EPSG
        self.depth_main_path=depth_main_path
        self.city=city
        self.main_url=main_url
        self.token=token

        self.json_log_path = LogUtils.log_path 
        
    def depth_of_pixel_navvis(self,pixel_data, x, y):
        pass

    def depth_of_pixel_cyclomedia(self,pixel_data, x, y):
        red = pixel_data[:, :, 2] / 255.0
        green = pixel_data[:, :, 1] / 255.0

        precision_bits = math.floor(red[math.floor(y), math.floor(x)] * 3.984375)
        depth_bits = (((red[math.floor(y), math.floor(x)] * 255.0) - (precision_bits * 64.0))) * 256.0 + (green[math.floor(y), math.floor(x)] * 255.0)
        precision_scale = math.pow(2.0, precision_bits * 2.0)
        
        depth_in_mm = depth_bits * precision_scale
        return depth_in_mm / 1000.0

    def depth_of_pixel_kgm(self,pixel_data, x, y):
        pass

    def global_coordinates(self,img_path, depth_path, pxi, pyi, ShootingPoint_latitude, ShootingPoint_longitude, ShootingPoint_height, source_epsg, data_type):
        pixel_data = cv2.imread(depth_path)
        if data_type == "Navvis":
            depth_px_py = self.depth_of_pixel_navvis(pixel_data, pxi, pyi)
            depth_center = 768
        elif data_type == 'Cyclomedia-Equirectangle-Yeni':
            depth_px_py = self.depth_of_pixel_cyclomedia(pixel_data, pxi, pyi)
            depth_center = 768
            recorder_direction = img_path.split("\\")[-1].split(".")[0].split("_")[-1]
            img_id = img_path.split("\\")[-1].split(".")[0]
        elif data_type == "KGM":
            depth_px_py = self.depth_of_pixel_kgm(pixel_data, pxi, pyi)
            depth_center = 768
        else:
            raise ValueError("Invalid data type provided.")
        
        if recorder_direction == 'F': 
            angle_x_adj = 90
        elif recorder_direction == 'B': 
            angle_x_adj = -90
        elif recorder_direction == 'L':  
            angle_x_adj = 180
        elif recorder_direction == 'R': 
            angle_x_adj = 0
        else:
            angle_x_adj = -90  

        angle_x = math.atan((768 - pxi) / depth_center) + math.radians(angle_x_adj)
        angle_y = math.atan((768 - pyi) / depth_center)

        utm_x, utm_y = self.latlon_to_utm(ShootingPoint_latitude, ShootingPoint_longitude, '4326', source_epsg)
        
        new_utm_x = utm_x + (depth_px_py * math.cos(angle_x))
        new_utm_y = utm_y + (depth_px_py * math.sin(angle_x))
        z = ShootingPoint_height + depth_px_py * math.sin(angle_y)
        new_lat, new_lon = self.utm_to_latlon(new_utm_x, new_utm_y, source_epsg, '4326')

        return {"Image_ID": img_id, "x": new_utm_x, "y": new_utm_y, "z": z, "lat": new_lat, "lon": new_lon}

    def latlon_to_utm(self,lat, lon, sourceepsg, targetepsg):
        pyproj_source = pyproj.CRS("EPSG:" + str(sourceepsg))
        pyproj_target = pyproj.CRS("EPSG:" + str(targetepsg))
        transformer = pyproj.Transformer.from_crs(pyproj_source, pyproj_target, always_xy=True)
        utm_x, utm_y = transformer.transform(lon, lat)
        return utm_x, utm_y

    def utm_to_latlon(self,utm_x, utm_y, sourceepsg, targetepsg):
        pyproj_source = pyproj.CRS("EPSG:" + str(sourceepsg))
        pyproj_target = pyproj.CRS("EPSG:" + str(targetepsg))
        transformer = pyproj.Transformer.from_crs(pyproj_source, pyproj_target, always_xy=True)
        lon, lat = transformer.transform(utm_x, utm_y)
        return lat, lon

    def main_global_coordinate_calculation(self, xml_data, depth_main_path,source_EPSG, data_type,img_id,boundingbox, min_height, max_height):

        if data_type == 'Cyclomedia-Cubemap-Karo':
            img_8id= img_id.split('_')[0]
            ShootingPoint_latitude = xml_data[xml_data['ImageId'] == img_8id]['Latitude'].values[0]
            ShootingPoint_longitude = xml_data[xml_data['ImageId'] == img_8id]['Longitude'].values[0]
            ShootingPoint_height = xml_data[xml_data['ImageId'] == img_8id]['Height'].values[0]
            point = f"POINT Z({ShootingPoint_longitude} {ShootingPoint_latitude} {ShootingPoint_height})"
            isFP=None
            return point, isFP
        
        elif data_type == 'Cyclomedia-Equirectangle-Yeni':
            img_8id= img_id.split('_')[0]
            ShootingPoint_latitude = xml_data[xml_data['ImageId'] == img_8id]['Latitude'].values[0]
            ShootingPoint_longitude = xml_data[xml_data['ImageId'] == img_8id]['Longitude'].values[0]
            ShootingPoint_height = xml_data[xml_data['ImageId'] == img_8id]['Height'].values[0]
            depth_path = os.path.join(depth_main_path, os.path.splitext(img_id)[0]+ '.png')
            if os.path.exists(depth_path) :
                x_center = (float(boundingbox[0]) + float(boundingbox[2])) / 2
                y_center = (float(boundingbox[1]) + float(boundingbox[3])) / 2
                result = self.global_coordinates(img_id, depth_path, x_center, y_center, ShootingPoint_latitude, ShootingPoint_longitude, ShootingPoint_height, source_EPSG, data_type)
                isFP = check_detection_fp_inthreshold(xml_data, result['lat'], result['lon'], result['z'], min_height, max_height, ShootingPoint_height)
                point = f"POINT Z({result['lon']} {result['lat']} {result['z']})"
                return point, isFP
            else:
                print(f"File path '{depth_path}' does not exist.")
                point = f"POINT Z({ShootingPoint_longitude} {ShootingPoint_latitude} {ShootingPoint_height})"
                isFP=None
                return point, isFP
            
        elif data_type == 'Leica':
            img_8id=img_id.split('_')[0]+'_'+img_id.split('_')[1]+'.jpg'
            ShootingPoint_X = xml_data[xml_data['Image filename'] == img_8id]['X1'].values[0]
            ShootingPoint_Y = xml_data[xml_data['Image filename'] == img_8id]['Y1'].values[0]
            ShootingPoint_Z = xml_data[xml_data['Image filename'] == img_8id]['Z1'].values[0]
            new_lat, new_lon = self.utm_to_latlon(ShootingPoint_X, ShootingPoint_Y, source_EPSG, '4326')
            point = f"POINT Z({new_lon} {new_lat} {ShootingPoint_Z})"
            isFP=None
            return point, isFP
        
        elif data_type == 'KGM':
            pass
        elif data_type == 'Navvis':
            pass
        else:
            raise ValueError("Invalid data type provided.")

    def find_img_files(self, main_directory):
        jpg_files = []
        for root, _, files in os.walk(main_directory):
            for file in files:
                if file.endswith('.jpg'):
                    jpg_files.append(os.path.join(root, file))
        return jpg_files

    def crop_images_with_bbox(self, out_tmp_dir,image_path, image,  bounding_boxes, classtr, classing,class_recid, fp_string, img_conf,class_dict_df):

        padding = 50
        cropped_image_dir = os.path.join(out_tmp_dir, 'Cropped_Images')

        if not os.path.exists(cropped_image_dir):
            os.makedirs(cropped_image_dir)

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found!!")
            return

        h, w = image.shape[:2]
        full_image_filename = os.path.basename(image_path)

        minx, miny, maxx, maxy = bounding_boxes
        class_id = classing
        class_filename = class_id.replace(' ', '')

        left = max(0, int(minx - padding))
        top = max(0, int(miny - padding))
        right = min(w, int(maxx + padding))
        bottom = min(h, int(maxy + padding))

        cropped_image = image[top:bottom, left:right].copy()
        bbox_color = (0, 0, 255) 
        bbox_thickness = 2
        cv2.rectangle(
            cropped_image,
            (int(minx - left), int(miny - top)),
            (int(maxx - left), int(maxy - top)),
            bbox_color,
            bbox_thickness
        )
        left_nopadding = max(0, int(minx))
        top_nopadding = max(0, int(miny))
        right_nopadding = min(w, int(maxx))
        bottom_nopadding = min(h, int(maxy))
        cropped_image_nopadding = image[top_nopadding:bottom_nopadding, left_nopadding:right_nopadding]

        if class_recid == "SpeedImage":
            try:
                resizer = SpeedImageResizer()
                resized_image = resizer.resize(cropped_image_nopadding)
                ocr_class_output = self.ocr_speed_model.process_image_array(resized_image)
                if ocr_class_output:
                    base_class_name = "Azami hız sınırı "
                    concatenated = f"{base_class_name}{ocr_class_output[0]}"
                    result_class_ocr = find_rec_id(class_dict_df, concatenated)
                    class_recid = result_class_ocr[0]
                    class_filename = result_class_ocr[2].replace(' ', '')
                else:
                    print("No text detected by Speed OCR.")
            except Exception as ocr_error:
                print(f"OCR Error: {ocr_error}")

        elif class_recid == "HeightImage": 
            try:
                resizer = HeightImageResizer()
                resized_image = resizer.resize(cropped_image_nopadding)
                ocr_class_output, raw_result = self.ocr_height_model.process_image_array(resized_image)
                if ocr_class_output:
                    max_height, valid_candidates = OCRUtils.get_max_height_sign(raw_result)
                    if max_height == "UNCLASSIFIED":
                        print("UNCLASSSIFIED !")
                    else:
                        print(f"Max Height: {max_height}")
                else:
                    print("UNDETECTED ! No text detected by Height OCR.")
            except Exception as ocr_error:
                print(f"Height OCR Error: {ocr_error}")


        if fp_string == "":
            cropped_filename = f"{os.path.splitext(full_image_filename)[0]}__{class_filename}__{str(img_conf).split('.')[1]}__{uuid.uuid4()}.jpg"
        else:
            cropped_filename = f"{os.path.splitext(full_image_filename)[0]}__{class_filename}__{str(img_conf).split('.')[1]}__{uuid.uuid4()}__FP.jpg"

        cropped_filepath = os.path.join(cropped_image_dir, cropped_filename)

        cv2.imwrite(cropped_filepath, cropped_image)

        return cropped_image, cropped_filename, class_recid

    def detect_objects(self, job_id, image_file,image_index, total_img_file, conf_values, image_size, save_img, save_txt, xml_data, depth_main_path, source_EPSG, data_type ,City, class_dict_df, main_url, token):
        output = []
        fp_string = ""
        percentage_complete = (image_index / total_img_file) * 100
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        try:
            results = self.model.predict(
                source=image_file,
                conf=conf_values,
                verbose=False,
                imgsz=image_size,
                save=save_img,
                line_width=1,
                save_txt=save_txt
            )
        except Exception as e:
            print(f"⚠️ Image Read Error: {image_file}")
            tb = traceback.format_exc()
            return output

        if results and results[0].boxes is not None and results[0].boxes.conf.numel() > 0:
            file_name = image_file.split("\\")[-1]

            for j in range(len(results[0].boxes)):
                img_conf = results[0].boxes.conf[j].item()
                if img_conf >= conf_values:
                    try:
                        class_id = results[0].boxes.cls[j].item()
                        class_name_yolo = results[0].names[class_id]

                        bounding_boxes = results[0].boxes.xyxy[j].tolist()
                        bounding_boxes_normalize = results[0].boxes.xywhn[j].tolist()
                        bounding_boxes_str = ','.join(map(str, bounding_boxes_normalize))
                        
                        result_class = find_rec_id(class_dict_df, class_name_yolo)

                        class_recid = result_class[0]
                        class_tr = result_class[1]
                        class_ing = result_class[2]
                        class_min_height = result_class[3]
                        class_max_height = result_class[4]
                        
                        source=' '
    
                        if data_type == 'Cyclomedia-Equirectangle-Yeni' or data_type == 'Cyclomedia-Cubemap-Karo':
                            Location, is_fp = self.main_global_coordinate_calculation(xml_data, depth_main_path, source_EPSG, data_type, image_id, bounding_boxes, class_min_height, class_max_height)
                            if is_fp is not None:
                                if is_fp['result'] == True:
                                    fp_string = "_FP_"
                                else:
                                    pass
                        else:
                            Location, is_fp = self.main_global_coordinate_calculation(xml_data, depth_main_path, source_EPSG, data_type, image_id, bounding_boxes, class_min_height, class_max_height)

                        image = results[0].orig_img
                        croped_data,croped_file_name,class_recid =self.crop_images_with_bbox(self.output_job_directory,image_file, image, bounding_boxes, class_tr, class_ing, class_recid, fp_string, img_conf,class_dict_df)
                        
                        AssetSubType = class_recid

                        Description = DescripitionBuilder.json_description_builder(image_id, xml_data, AssetSubType, data_type)
                        print(croped_file_name)
                        
                    except Exception as e:
                        print(f"Error processing image {image_id}: {e}")
                        tb = traceback.format_exc()
                        raise e

        else :
            if image_index == total_img_file :       
                print("Image Detection")
                pass
        return output

    def run(self):

        self.start_time = time.time()
        jpg_list = self.find_img_files(self.input_directories) if self.datatype != 'Cyclomedia-Cubemap-Karo' else [
            img for img in self.find_img_files(self.input_directories)
            if any(pattern in img for pattern in ['2_B_0_1', '2_B_1_1', '2_B_2_1', '2_F_0_1', '2_F_1_1', '2_F_2_1', '2_R_0_1', '2_R_1_1', '2_R_2_1', '2_L_0_1', '2_L_1_1', '2_L_2_1'])
        ]

        if not jpg_list:
            raise FileNotFoundError(f"No images found in {self.input_directories} (data type: {self.datatype}).")

        jpg_list.sort()

        start_from = LogUtils.resume_index(self.json_log_path, jpg_list,
                                   key="Detection / Segmentation")
        if start_from:
            print(f"[RESUME] Skipping first {start_from} already-done images.")
        
        values = ast.literal_eval(self.model_parameters)
        conf_values = float(values[0])
        image_size = int(values[1])
        save_img = values[2]
        save_txt = values[3]
        results = []

        City=self.city

        class_dict_filepath = os.path.join(os.path.dirname(__file__),"Detection_Classes.csv")
        class_dict_df = pd.read_csv(class_dict_filepath, encoding='utf-8')

        for index, image_file in enumerate(jpg_list[start_from:], start=start_from + 1):
            file_name = os.path.split(image_file)[-1]
            
            if self.datatype == 'Cyclomedia-Cubemap-Karo' or self.datatype == 'Cyclomedia-Equirectangle-Yeni':
                detected_id = file_name[:10]
            
            elif self.datatype == 'Leica':
                file_name_without_extension, file_extension = os.path.splitext(file_name)
                detected_id = file_name_without_extension

            else:
                detected_id = file_name.split(".")[0]
            
            Cropped_Images_path = os.path.join(self.output_job_directory, 'Cropped_Images')

            search_pattern = os.path.join(Cropped_Images_path, f'{detected_id}.jpg')
            existing_files = glob.glob(search_pattern)

            if existing_files:
                pass
            else:
                results.extend(self.detect_objects(
                    self.job_id, image_file, index + 1, len(jpg_list), conf_values,
                    image_size, save_img, save_txt, self.xml_data, self.depth_main_path,
                    self.source_EPSG, self.datatype, City, class_dict_df,
                    self.main_url, self.token))
                
                LogUtils.update_last_processed(self.json_log_path,
                               key="Detection / Segmentation",
                               idx=index + 1,
                               total=len(jpg_list),
                               filename=os.path.basename(image_file))
            _AsyncLastWriter.flush_now()