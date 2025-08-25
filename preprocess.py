import os
import xml_reader
import cv2
from PIL import Image
import utils
import numpy as np
import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import glob
from general_utils import LogUtils , _AsyncLastWriter

def main_preprocess(data_type, main_directory, num_process, job_id, main_url, token, source_EPSG,subdir_name,num_folders):

    xml_data, depth_directory, image_directory, xml_EPSG, data_sub_type = main_directory_check(data_type, main_directory, source_EPSG,subdir_name)
    
    preprocess_img_output_directory, depth_directory = preprocess(data_sub_type, xml_data, depth_directory, image_directory, main_directory, num_process, job_id, main_url, token,num_folders)
    print(preprocess_img_output_directory, depth_directory, xml_EPSG, data_sub_type)
    return preprocess_img_output_directory, depth_directory, xml_data,xml_EPSG, data_sub_type

def main_directory_check(data_type, main_directory, source_EPSG,subdir_name):
    if data_type == 'Cyclomedia':
        xml_data = check_xml_in_parent_directory(subdir_name)
        dataset_name = xml_data['Dataset_Name'][0]
        xml_EPSG = xml_data['SRID'].values[0]
        data_sub_type = getcycdatatype(main_directory,dataset_name)
        if data_sub_type == 'Cyclomedia-Equirectangle-Yeni':
            depth_directory=check_depth_karo_directories(main_directory,dataset_name)
            equirectangle_directory=check_equirectangle_directories(main_directory,dataset_name)
            return xml_data, depth_directory ,equirectangle_directory, xml_EPSG,data_sub_type
            
        elif data_sub_type == 'Cyclomedia-Cubemap-Karo':
            depth_directory= ''
            karo_directory=check_karo_directories(main_directory,dataset_name)
            return xml_data, depth_directory ,karo_directory, xml_EPSG,data_sub_type
    
    elif data_type == 'Navvis':
        pass  

    elif data_type == 'KGM':
        pass  
    
    elif data_type == 'Leica':
        dataset_name='360photos'
        xml_EPSG=source_EPSG
        data_sub_type='Leica'
        xml_data = check_csv_in_parent_directory(os.path.join(main_directory,dataset_name))
        depth_directory=''
        equirectangle_directory=check_equirectangle_directories(main_directory,dataset_name)
        return xml_data, depth_directory ,equirectangle_directory, xml_EPSG, data_sub_type
    
def getcycdatatype(directory, dataset_name):
    cycimg_directory = os.path.join(directory, os.path.normcase(dataset_name))
    if not os.path.exists(cycimg_directory):
        print(f"Not found directory: {cycimg_directory}")
        return

    has_depth_folder = False
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)) and 'depth' in subdir.lower():
            has_depth_folder = True
            break

    if has_depth_folder:
        data_type = 'Cyclomedia-Equirectangle-Yeni'
    else:
        data_type = 'Cyclomedia-Cubemap-Karo'

    return data_type

def check_equirectangle_directories(directory,dataset_name):
    equirectangle_directory = os.path.join(directory,os.path.normcase(dataset_name))
    if os.path.isdir(equirectangle_directory):
        print("Equirectangle available.")
    else:
        raise ValueError("No equirectangle files!")
    return equirectangle_directory

def check_karo_directories(directory,dataset_name):
    equirectangle_directory = os.path.join(directory,os.path.normcase(dataset_name))
    if os.path.isdir(equirectangle_directory):
        print("Karo available.")
    else:
        raise ValueError("No Karo files!")
    return equirectangle_directory

def check_depth_karo_directories(directory, dataset_name):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    depth_directory = None
    for subdir in subdirectories:
        subdir_lower = subdir.lower()
        if 'depth' in subdir_lower and subdir_lower != 'merged_depth_cubefaces_output' and subdir_lower != 'merged_depth_cubefaces_output_completed':
            depth_directory = os.path.join(directory, subdir)
            break

    if depth_directory:
        potential_depth_path = os.path.join(depth_directory, dataset_name)
        if os.path.isdir(potential_depth_path):
            depth_path = potential_depth_path
        else:
            depth_path = depth_directory
        print(f"Depth available at: {depth_path}")
    else:
        raise ValueError("No directory containing 'depth' found!")
    return depth_path

def check_xml_in_parent_directory(subdir):
    
    parent_directory = os.path.dirname(subdir)

    directory_name = os.path.basename(subdir)
    base_name = directory_name.split('.')[0]
        
    xml_files = [file for file in os.listdir(parent_directory) if file.endswith('.xml') and file.startswith(base_name)]
        
    if xml_files:
        xml_file_path = os.path.join(parent_directory, xml_files[0])
        xml_data = xml_reader.get_lat_lon_height_from_xml(xml_file_path)
    else:
        raise ValueError(f"No matching XML file found in parent directory starting with '{base_name}'.")
    

    return xml_data

def check_csv_in_parent_directory(directory):
    xml_folder = [file for file in os.listdir(directory) if file.endswith('.csv')]
    if  len(xml_folder) > 0:
        xml_path=os.path.join(directory,xml_folder[0])
        xml_data = pd.read_csv(xml_path, delimiter=';')
    else:
        raise ValueError("No xml file!")
    return xml_data

def list_all_imgs_depths(main_directory):
    jpg_files = []
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith('.jpeg'):
                full_path = os.path.join(root, file)
                jpg_files.append(full_path)
    return jpg_files

def preprocess(data_type, xml_data, depth_original_path, img_original_path, main_directory, num_process,job_id,main_url,token,num_folders):
    if data_type == 'Cyclomedia-Equirectangle-Yeni':
        all_imgs_files = list_all_imgs_depths(img_original_path)
        total_imgs_count = len(all_imgs_files)

        all_depths_files = list_all_imgs_depths(depth_original_path)
        total_depth_count = len(all_depths_files)

        merged_depth_cubefaces_output_path = os.path.join(main_directory, 'merged_depth_cubefaces_output')
        merged_depth_completed_path = os.path.join(main_directory, 'merged_depth_cubefaces_output_completed')
        if num_folders > 1:
            try:
                os.makedirs(merged_depth_cubefaces_output_path)
            except Exception:
                pass
            merge_depth_tiles(data_type, all_depths_files, merged_depth_cubefaces_output_path, num_process, total_depth_count, job_id, main_url, token)
        else:    
            if not os.path.exists(merged_depth_completed_path):
                if not os.path.exists(merged_depth_cubefaces_output_path):
                    os.makedirs(merged_depth_cubefaces_output_path)

                merge_depth_tiles(data_type, all_depths_files, merged_depth_cubefaces_output_path, num_process, total_depth_count, job_id, main_url, token)

                os.rename(merged_depth_cubefaces_output_path, merged_depth_completed_path)

        splited_cubemap_img_output_path = os.path.join(main_directory, 'splited_cubemap_img_output')
        splited_cubemap_img_completed_path = os.path.join(main_directory, 'splited_cubemap_img_output_completed')
        
        splited_cubemap_karo_img_output_path = os.path.join(main_directory, 'splited_cubemap_karo_img_output')
        splited_cubemap_karo_img_completed_path = os.path.join(main_directory, 'splited_cubemap_karo_img_output_completed')

        if num_folders > 1:         
            convert_type = "e2c"
            width = 1536
            height = 1536
            try:
                os.makedirs(splited_cubemap_img_output_path)
                os.makedirs(splited_cubemap_karo_img_output_path)  
            except Exception:
                pass
            panorama_convert_and_split(data_type, all_imgs_files, splited_cubemap_img_output_path, convert_type, width, height, splited_cubemap_karo_img_output_path, num_process, total_imgs_count, job_id, main_url, token)
        else:    
            if not os.path.exists(splited_cubemap_img_completed_path):
                convert_type = "e2c"
                width = 1536
                height = 1536

                if not os.path.exists(splited_cubemap_img_output_path):
                   os.makedirs(splited_cubemap_img_output_path)
            
                if not os.path.exists(splited_cubemap_karo_img_output_path):
                    os.makedirs(splited_cubemap_karo_img_output_path)

                panorama_convert_and_split(data_type, all_imgs_files, splited_cubemap_img_output_path, convert_type, width, height, splited_cubemap_karo_img_output_path, num_process, total_imgs_count, job_id, main_url, token)

                os.rename(splited_cubemap_img_output_path, splited_cubemap_img_completed_path)
                os.rename(splited_cubemap_karo_img_output_path, splited_cubemap_karo_img_completed_path)

        return splited_cubemap_karo_img_completed_path, merged_depth_completed_path
        
    elif data_type == 'Cyclomedia-Cubemap-Karo':

        merged_depth_completed_path=''
        
        splited_cubemap_karo_img_completed_path = img_original_path

        return splited_cubemap_karo_img_completed_path, merged_depth_completed_path

    elif data_type == 'Navvis':
        pass

    elif data_type == 'KGM':
        pass
   
    elif data_type == 'Leica':
        
        all_imgs_files = list_all_imgs_depths(img_original_path)
        total_imgs_count = len(all_imgs_files)

        merged_depth_completed_path=''
        
        splited_cubemap_img_output_path = os.path.join(main_directory, 'splited_cubemap_img_output')
        splited_cubemap_img_completed_path = os.path.join(main_directory, 'splited_cubemap_img_output_completed')
        
        splited_cubemap_karo_img_output_path = os.path.join(main_directory, 'splited_cubemap_karo_img_output')
        splited_cubemap_karo_img_completed_path = os.path.join(main_directory, 'splited_cubemap_karo_img_output_completed')

        if not os.path.exists(splited_cubemap_img_completed_path):
            convert_type = "e2c"
            width = 1536
            height = 1536

            if not os.path.exists(splited_cubemap_img_output_path):
                os.makedirs(splited_cubemap_img_output_path)
            
            if not os.path.exists(splited_cubemap_karo_img_output_path):
                os.makedirs(splited_cubemap_karo_img_output_path)

            panorama_convert_and_split(data_type, all_imgs_files, splited_cubemap_img_output_path, convert_type, width, height, splited_cubemap_karo_img_output_path, num_process, total_imgs_count, job_id, main_url, token)

            os.rename(splited_cubemap_img_output_path, splited_cubemap_img_completed_path)
            os.rename(splited_cubemap_karo_img_output_path, splited_cubemap_karo_img_completed_path)

        return splited_cubemap_karo_img_completed_path, merged_depth_completed_path

def merge_depth_tiles(data_type, all_depth_directory, output_directory, thread_count, total_process_img, job_id, main_url, token):

    all_depth_directory = sorted(all_depth_directory)
    start_from = LogUtils.resume_index(LogUtils.log_path, all_depth_directory, key= "Depth Preprocess")
    if start_from:
        print(f"[RESUME] Skipping first {start_from} depth tiles")

    logical_cores = psutil.cpu_count(logical=True)

    num_threads = min(thread_count, logical_cores)

    threads = []
    
    for i, filename in enumerate(all_depth_directory[start_from:], start=start_from):
        
        thread = threading.Thread(target=process_depth_tile, args=((filename, i, data_type, output_directory, total_process_img, job_id, main_url, token),))
        threads.append(thread)
        thread.start()

        if len(threads) >= num_threads:
            for thread in threads:
                thread.join()
            threads = []

    for thread in threads:
        thread.join()

    _AsyncLastWriter.flush_now()

def process_depth_tile(args):
    filename, image_index, data_type, output_directory, total_process_img, job_id, main_url, token = args
    codes = ['F', 'B', 'R', 'L']
    try:
        if data_type == 'Cyclomedia-Equirectangle-Yeni':
            image_id = os.path.splitext(os.path.basename(filename))[0][:8]
            image_directory_path = os.path.dirname(filename)
        else:
            image_id = os.path.splitext(os.path.basename(filename))[0]
            image_directory_path = os.path.dirname(filename)

        if all([os.path.isfile(os.path.join(output_directory, f"{image_id}_{code}.png")) for code in codes]):
            
            LogUtils.update_last_processed(
                LogUtils.log_path,
                key="Depth Preprocess",
                idx=image_index + 1,               
                total=total_process_img,
                filename=f"{image_id}_{codes[-1]}.png",  
            )
            progress = (image_index + 1) / total_process_img * 100   
            print(progress)                                          
            return

        for idx in codes:
            image00 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_0_0.png"))
            image10 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_1_0.png"))
            image20 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_2_0.png"))
            panorama0 = cv2.hconcat([image00, image10, image20])                        

            image01 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_0_1.png"))
            image11 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_1_1.png"))
            image21 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_2_1.png"))
            panorama1 = cv2.hconcat([image01, image11, image21])

            image02 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_0_2.png"))
            image12 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_1_2.png"))
            image22 = cv2.imread(os.path.join(image_directory_path, f"{image_id}_2_{idx}_2_2.png"))
            panorama2 = cv2.hconcat([image02, image12, image22])                        

            final_panorama = cv2.vconcat([panorama0, panorama1, panorama2])
            output_path = os.path.join(output_directory, f"{image_id}_{idx}.png")
            cv2.imwrite(output_path, final_panorama)

            progress = (image_index + 1) / total_process_img * 100 
            print(progress)

            LogUtils.update_last_processed(
                LogUtils.log_path,
                idx=image_index + 1,
                total=total_process_img,
                filename=f"{image_id}_{idx}.png",
                key="Depth Preprocess"
            )

            print(f"{image_id}_{idx}.png", "Preprocess-Depth")
            
    except Exception as e:
        tb = traceback.format_exc()
        progress = (image_index / total_process_img) * 100
        raise e

def panorama_convert_and_split(data_type, input_folder, output_folder, convert_type, width, height, split_output, thread_count,total_process_img,job_id,main_url,token):

    logical_cores = psutil.cpu_count(logical=True)

    num_threads = min(thread_count, logical_cores)

    threads = []

    input_folder = sorted(input_folder)
    start_from = LogUtils.resume_index(LogUtils.log_path, input_folder, key= "Image Preprocess")
    if start_from:
        print(f"[RESUME] Skipping first {start_from} panoramas")

    for i, filename in enumerate(input_folder[start_from:], start=start_from):
        thread = threading.Thread(target=panorama_process_files, args=((filename, i, output_folder, convert_type, width, height, split_output, total_process_img, job_id, main_url, token),))
        threads.append(thread)
        thread.start()

        if len(threads) >= num_threads:
            for thread in threads:
                thread.join()
            threads = []

    for thread in threads:
        thread.join()

    _AsyncLastWriter.flush_now()
    
def panorama_process_files(args):
    pano_filename, i, output_folder, convert_type, width, height, split_output, total_process_img, job_id, main_url, token = args
    
    panorama_imgid = os.path.basename(pano_filename).split('.')[0]
        
    output_path = os.path.join(output_folder, os.path.basename(pano_filename))

    if os.path.isfile(output_path):
        pass
    else:
        h_fov = 60
        v_fov = 60
        u_deg = 0
        v_deg = 0
        in_rot_deg = 0
        img = np.array(Image.open(pano_filename))
        
        if len(img.shape) == 2:
            img = img[..., None]
        if convert_type == 'c2e':
            out = utils.c2e(img, h=height, w=width, mode='bilinear')
        elif convert_type == 'e2c':
            out = utils.e2c(img, face_w=width, mode='bilinear')
        elif convert_type == 'e2p':
            out = utils.e2p(img, fov_deg=(h_fov, v_fov), u_deg=u_deg, v_deg=v_deg,
                            out_hw=(height, width), in_rot_deg=in_rot_deg, mode='bilinear')
        else:
            raise NotImplementedError('Unknown conversion')
        
        processed_num_img = i

        panorama_split_image(out,panorama_imgid, split_output,processed_num_img, total_process_img, job_id, main_url, token)

        LogUtils.update_last_processed(
            LogUtils.log_path,
            idx=processed_num_img + 1,
            total=total_process_img,
            filename=os.path.basename(pano_filename),
            key="Image Preprocess"
        )
                
def panorama_split_image(image_array,panorama_imgid, output_path, processed_num_img, total_process_img, job_id, main_url, token): 
    
    height, width = image_array.shape[:2]
    

    part_width = width // 4
    part_height = height // 3

    regions = [
        (part_height, 2 * part_height, part_width, 2 * part_width),
        (part_height, 2 * part_height, 2 * part_width, 3 * part_width),
        (part_height, 2 * part_height, 0, part_width),
        (part_height, 2 * part_height, 3 * part_width, 4 * part_width),
    ]

    direction = ['F', 'R', 'L', 'B']
    
    for i, (y_start, y_end, x_start, x_end) in enumerate(regions):
        crop = image_array[y_start:y_end, x_start:x_end]
        panorama_img = os.path.join(output_path, f"{panorama_imgid}_{direction[i]}.jpg")
        cv2.imwrite(panorama_img, crop[..., ::-1])  
        process = (processed_num_img / total_process_img) * 100
        print(process)
        print(f"{panorama_imgid}_{direction[i]}.jpg", "Preprocess-Image")