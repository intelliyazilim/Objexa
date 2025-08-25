import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import functools
from pathlib import Path
from typing import Any, Union
import shutil
import cv2
import os
import math
import pyproj
import base64
from general_utils import DescripitionBuilder

def log_execution_time(process_name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = process_name or func.__name__ 
            print(f"{name} process has been started.")
            start_time = time.time()
            
            result = func(*args, **kwargs) 
            
            end_time = time.time()
            print(f"{name} process has ended.")
            print(f"Total execution time of {name}: {end_time - start_time:.2f} seconds.")
            
            return result
        return wrapper
    return decorator

def depth_of_pixel_navvis(self,pixel_data, x, y):
    pass

def depth_of_pixel_kgm(self,pixel_data, x, y):
    pass

def utm_to_latlon(utm_x, utm_y, sourceepsg, targetepsg):

    pyproj_source = pyproj.CRS("EPSG:" + str(sourceepsg))
    pyproj_target = pyproj.CRS("EPSG:" + str(targetepsg))
    transformer = pyproj.Transformer.from_crs(pyproj_source, pyproj_target, always_xy=True)
    lon, lat = transformer.transform(utm_x, utm_y)
    return lat, lon

def latlon_to_utm(lat, lon, sourceepsg, targetepsg):

    pyproj_source = pyproj.CRS("EPSG:" + str(sourceepsg))
    pyproj_target = pyproj.CRS("EPSG:" + str(targetepsg))
    transformer = pyproj.Transformer.from_crs(pyproj_source, pyproj_target, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y

def depth_of_pixel_cyclomedia(pixel_data, x, y):

    red = pixel_data[:, :, 2] / 255.0
    green = pixel_data[:, :, 1] / 255.0
    precision_bits = math.floor(red[math.floor(y), math.floor(x)] * 3.984375)
    depth_bits = (((red[math.floor(y), math.floor(x)] * 255.0) - (precision_bits * 64.0))) * 256.0 + (green[math.floor(y), math.floor(x)] * 255.0)
    precision_scale = math.pow(2.0, precision_bits * 2.0)
    
    depth_in_mm = depth_bits * precision_scale
    return depth_in_mm / 1000.0

def global_coordinates(img_path, depth_path, pxi, pyi, ShootingPoint_latitude, ShootingPoint_longitude, ShootingPoint_height, source_epsg, data_type):

    pixel_data = cv2.imread(depth_path)
    if data_type == "Navvis":
        depth_px_py = depth_of_pixel_navvis(pixel_data, pxi, pyi)
        depth_center = 768
    elif data_type == 'Cyclomedia-Equirectangle-Yeni':
        depth_px_py = depth_of_pixel_cyclomedia(pixel_data, pxi, pyi)
        depth_center = 768
        recorder_direction = img_path.split("\\")[-1].split(".")[0].split("_")[-1]
        img_id = img_path.split("\\")[-1].split(".")[0]
    elif data_type == "KGM":
        depth_px_py = depth_of_pixel_kgm(pixel_data, pxi, pyi)
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
    utm_x, utm_y = latlon_to_utm(ShootingPoint_latitude, ShootingPoint_longitude, '4326', source_epsg)
    
    new_utm_x = utm_x + (depth_px_py * math.cos(angle_x))
    new_utm_y = utm_y + (depth_px_py * math.sin(angle_x))
    z = ShootingPoint_height + depth_px_py * math.sin(angle_y)
    new_lat, new_lon = utm_to_latlon(new_utm_x, new_utm_y, source_epsg, '4326')
    return {"Image_ID": img_id, "x": new_utm_x, "y": new_utm_y, "z": z, "lat": new_lat, "lon": new_lon}

def main_global_coordinate_calculation(xml_data, depth_main_path,source_EPSG, data_type,img_id,boundingbox):

        if data_type == 'Cyclomedia-Cubemap-Karo':
            img_8id= img_id.split('_')[0]
            ShootingPoint_latitude = xml_data[xml_data['ImageId'] == img_8id]['Latitude'].values[0]
            ShootingPoint_longitude = xml_data[xml_data['ImageId'] == img_8id]['Longitude'].values[0]
            ShootingPoint_height = xml_data[xml_data['ImageId'] == img_8id]['Height'].values[0]
            point = f"POINT Z({ShootingPoint_longitude} {ShootingPoint_latitude} {ShootingPoint_height})"

            return point
        
        elif data_type == 'Cyclomedia-Equirectangle-Yeni':
            img_8id= img_id.split('_')[0]
            ShootingPoint_latitude = xml_data[xml_data['ImageId'] == img_8id]['Latitude'].values[0]
            ShootingPoint_longitude = xml_data[xml_data['ImageId'] == img_8id]['Longitude'].values[0]
            ShootingPoint_height = xml_data[xml_data['ImageId'] == img_8id]['Height'].values[0]
            depth_path = os.path.join(depth_main_path, os.path.splitext(img_id)[0]+ '.png')
            if os.path.exists(depth_path) :
                x_center = (float(boundingbox[0]) + float(boundingbox[2])) / 2
                y_center = (float(boundingbox[1]) + float(boundingbox[3])) / 2
                result = global_coordinates(img_id, depth_path, x_center, y_center, ShootingPoint_latitude, ShootingPoint_longitude, ShootingPoint_height, source_EPSG, data_type)
                point = f"POINT Z({result['lon']} {result['lat']} {result['z']})"

                return point
            else:
                print(f"File path '{depth_path}' does not exist.")
                point = f"POINT Z({ShootingPoint_longitude} {ShootingPoint_latitude} {ShootingPoint_height})"

                return point
            
        else:
            img_8id=img_id.split('_')[0]+'_'+img_id.split('_')[1]+'.jpg'
            ShootingPoint_X = xml_data[xml_data['Image filename'] == img_8id]['X1'].values[0]
            ShootingPoint_Y = xml_data[xml_data['Image filename'] == img_8id]['Y1'].values[0]
            ShootingPoint_Z = xml_data[xml_data['Image filename'] == img_8id]['Z1'].values[0]
            new_lat, new_lon = utm_to_latlon(ShootingPoint_X, ShootingPoint_Y, source_EPSG, '4326')
            point = f"POINT Z({new_lon} {new_lat} {ShootingPoint_Z})"

            return point

def crop_and_annotate(
    pil_img,
    mask,
    class_name,
    *,
    padding: int = 20,
    draw_mask: bool = False,
    draw_banner: bool = False,
    mask_alpha: float = 0.3,
    encode_jpeg: bool = True,
):

    ys, xs = np.where(mask > 0)
    if ys.size == 0:                                     
        return None, None, None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    h, w = mask.shape
    x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
    x2, y2 = min(x2 + padding, w - 1), min(y2 + padding, h - 1)


    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    crop_mask = mask[y1:y2, x1:x2]
    crop_rgba = crop.convert("RGBA")                 


    if draw_mask:
        solid_blue = Image.new("RGB", crop.size, (0, 0, 255))
        blended = Image.blend(crop, solid_blue, mask_alpha)
        mask_img = Image.fromarray((crop_mask * 255).astype("uint8"), mode="L")
        crop_rgba = Image.composite(blended, crop, mask_img).convert("RGBA")


    cnts, _ = cv2.findContours(
        (crop_mask * 255).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    draw = ImageDraw.Draw(crop_rgba)
    for cnt in cnts:
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        if len(pts) > 1:
            draw.line(pts + [pts[0]], fill=(0, 0, 255, 255), width=2)

    if draw_banner:
        margin_x, margin_y = 10, 5
        banner_h = max(int(0.2 * crop_rgba.height), margin_y * 2 + 10)
        font_sz = max(banner_h - 2 * margin_y, 14)
        try:
            font = ImageFont.truetype("arial.ttf", font_sz)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_sz)
            except IOError:
                font = ImageFont.load_default()

        dummy = ImageDraw.Draw(crop_rgba)
        try:
            text_w, text_h = font.getsize(class_name)
        except AttributeError:  
            bbox = dummy.textbbox((0, 0), class_name, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        pad_x = max((text_w - crop_rgba.width) // 2 + margin_x, 0)
        new_w = crop_rgba.width + 2 * pad_x
        new_h = crop_rgba.height + banner_h

        canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 255))
        canvas.paste(crop_rgba, (pad_x, 0))

        banner_y = crop_rgba.height
        draw_canvas = ImageDraw.Draw(canvas)
        draw_canvas.rectangle([(0, banner_y), (new_w, new_h)], fill=(0, 0, 255, 255))
        text_x = (new_w - text_w) // 2
        text_y = banner_y + (banner_h - text_h) // 2
        draw_canvas.text((text_x, text_y), class_name, font=font,
                         fill=(255, 255, 255, 255))
        final_rgb = canvas.convert("RGB")
    else:
        final_rgb = crop_rgba.convert("RGB")

    arr_rgb = np.array(final_rgb)       

    if encode_jpeg:
        success, buf = cv2.imencode(".jpg", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("JPEG encoding failed.")
        b64 = base64.b64encode(buf).decode("utf-8")
    else:
        b64 = None

    return final_rgb, b64, arr_rgb

def save_results(result: Any, image: Union[str, Path, np.ndarray, Image.Image], output_folder: Union[str, Path],
    *,
    save_txt: bool = True,
    save_crops: bool = True,
    save_image: bool = True,
    save_conf: bool = True,
    suffix: str = "") -> None:
    
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)


    if isinstance(image, (str, Path)):
        base = Path(image).stem
    elif hasattr(result, "path"):
        base = Path(result.path).stem
    else:
        base = "output"
    stem = f"{base}{suffix}"

    if save_txt:
        labels_dir = out_dir / "labels_txt"
        labels_dir.mkdir(exist_ok=True)
        txt_file = labels_dir / f"{stem}.txt"
        result.save_txt(str(txt_file), save_conf)

    if save_crops:
        result.save_crop(str(out_dir))

    if save_image:
        infer_dir = out_dir / "inference"
        infer_dir.mkdir(exist_ok=True)

        if isinstance(image, (str, Path)):
            src = Path(image)
            if not src.exists():
                raise FileNotFoundError(f"Input image not found: {src}")
            dest = infer_dir / f"{stem}{src.suffix}"
            shutil.copy(src, dest)

        elif isinstance(image, Image.Image):
            ext = ".png" if image.mode == "RGBA" else ".jpg"
            out_path = infer_dir / f"{stem}{ext}"
            image.save(out_path)

        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            out_path = infer_dir / f"{stem}.jpg"
            pil_img.save(out_path)

        else:
            raise TypeError("`image` must be a path, numpy array, or PIL.Image.")

def detection_details_wrapper(filename, xml_data, City, Location, class_recid, prob, mask, bbox, datatype):

    source = ""

    bounding_boxes = ','.join(map(str, bbox))
    Description =  DescripitionBuilder.json_description_builder(filename.split('.')[0], xml_data, class_recid, datatype)
    
    print(Description)
    