from general_utils import LogUtils, DecisionUtils
import traceback
import preprocess
import yolov8_detector
import argparse
import time
import os
import glob
from pathlib import Path
from segmentation import ImageSegmentation

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--process_type', default='full', type=str,  choices=['full', 'preprocess'])
        parser.add_argument('--data_type', type=str, required=True, choices=['Cyclomedia', 'Navvis', 'KGM', 'Leica', 'Others'], help='Veri tipini belirtin (Cyclomedia, Navvis, KGM)')
        parser.add_argument('--main_directory', type=str, required=True, help='Main Directory')
        parser.add_argument('--num_process', type=str, required=True, help='Multiprocessing parameter')
        parser.add_argument("--source_EPSG", type=str, required=True, help="EPSG Code")    
        parser.add_argument('--model_weights', type=str, required=True, help='Path to YOLO model')
        parser.add_argument('--model_parameters', type=str, required=True, help='YOLO model weights')
        parser.add_argument("--city_id", type=str, required=True, help="RecID of the City")
        parser.add_argument('--job_id', type=str, required=True, help='Job ID')
        parser.add_argument("--main_url", required=True, type=str, help="main_url")
        parser.add_argument('--token', required=True, type=str, help='token')
        args = parser.parse_args()
        
        start_time = time.time() 

        if not args.main_directory or not isinstance(args.main_directory, str):
            raise ValueError(f"Geçersiz 'main_directory' parametresi alındı: {repr(args.main_directory)}")

        out_directory_directory=os.path.join(Path(__file__).parent,'tmp')
        output_job_directory = os.path.join(out_directory_directory, args.job_id)

        os.makedirs(out_directory_directory, exist_ok=True)
        os.makedirs(output_job_directory, exist_ok=True)

        LogUtils.log_path = os.path.join(output_job_directory , args.job_id + '.json')

        if not os.path.exists(args.main_directory):
            raise FileNotFoundError(f"Main directory '{args.main_directory}' not found.")

        if not os.path.exists(LogUtils.log_path):
            LogUtils.create(
                LogUtils.log_path,
                process_id=args.job_id,
                data_type=args.data_type,
                output_directory=args.main_directory,
                model_weights=args.model_weights,
                model_parameters=args.model_parameters,
                start_date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                depth_preprocess=0,
                image_preprocess=0,
                detection_segmentation=0,
            )
        LogUtils.init_async_writer(LogUtils.log_path, interval=1.0)

        if args.data_type != 'Others':

            numprocess = int(args.num_process)

            if args.data_type == 'Cyclomedia':
                xml_files = glob.glob(os.path.join(args.main_directory, "*.xml"))
                if not xml_files:
                    raise FileNotFoundError(f"No XML files found in '{args.main_directory}'.")
                
                subdirectories = [
                    os.path.join(args.main_directory, subdir.upper())
                    for subdir in os.listdir(args.main_directory)
                    if os.path.isdir(os.path.join(args.main_directory, subdir.upper())) and 
                    any(xml.startswith(os.path.join(args.main_directory, subdir.upper())) for xml in xml_files)
                ]
            elif args.data_type == 'Leica':
                subdirectories = [args.main_directory]

            if not subdirectories:
                raise FileNotFoundError(f"No valid subdirectories with files in '{args.main_directory}'.")
            
            print('main_image_detection')
            print('Imagedan preprocess işlemleri başlatılıyor..')

            print("Starting preprocessing for all subdirectories...")
            preprocess_results = []  
            num_folders= len(subdirectories)    

            for subdir in subdirectories: 
                try:
                    print(f"Preprocessing subdirectory: {subdir}")
                    preprocess_img_output_directory, depth_path, xml_data, xml_EPSG, data_sub_type = preprocess.main_preprocess(
                        args.data_type, args.main_directory, numprocess, args.job_id, args.main_url, args.token, args.source_EPSG, subdir,num_folders)

                    preprocess_results.append({
                        'subdir': subdir,
                        'preprocess_img_output_directory': preprocess_img_output_directory,
                        'depth_path': depth_path,
                        'xml_data': xml_data,
                        'xml_EPSG': xml_EPSG,
                        'data_sub_type': data_sub_type,
                    })
                    print(f"Completed preprocessing for: {subdir}")
                except Exception as e:
                    print(f"Error during preprocessing of '{subdir}': {e}")
                    traceback.print_exc()
            end_time = time.time()
            elapsed_time = end_time - start_time 
            print(f"Preprocess işlemi tamamlandı. Geçen süre: {elapsed_time:.2f} saniye.")

            print('Imagedan preprocess işlemleri tamamlandı..')

            if len(subdirectories)==1:
                placeholder_num=1
            else:
                splited_cubemap_karo_img_output_path = os.path.join(args.main_directory, 'splited_cubemap_karo_img_output')
                splited_cubemap_img_output_path = os.path.join(args.main_directory, 'splited_cubemap_img_output')
                
                splited_cubemap_img_completed_path = os.path.join(args.main_directory, 'splited_cubemap_img_output_completed')
                splited_cubemap_karo_img_completed_path = os.path.join(args.main_directory, 'splited_cubemap_karo_img_output_completed')
                try:
                    os.rename(splited_cubemap_img_output_path, splited_cubemap_img_completed_path)
                    os.rename(splited_cubemap_karo_img_output_path, splited_cubemap_karo_img_completed_path)
                except Exception as e:
                    tb = traceback.format_exc()
                    raise e
                
            if args.process_type == 'preprocess':

                print("Preprocess işlemi tamamlandı.")
                exit(0)

            process = DecisionUtils.decide_process_type(args.model_weights)

            for result in preprocess_results:
                subdir = result['subdir']
                preprocess_img_output_directory=result['preprocess_img_output_directory']
                xml_file = result['xml_data']

                if process == 'segmentation':
                
                    print('Imagedan segmentasyon işlemleri başlatılıyor..')

                    segmentation = ImageSegmentation(input_folder=preprocess_img_output_directory,
                        output_folder=preprocess_img_output_directory, model_path=args.model_weights,
                        job_id=args.job_id,city_id=args.city_id, main_url=args.main_url, token=args.token,
                        xml_data=xml_file, depth_main_path=depth_path, source_EPSG=args.source_EPSG, data_type=data_sub_type,
                        save_img=True, save_txt=True, save_crops=True)

                    segmentation.load_model()
                    segmentation.process_images(padding=20, mask_alpha=0.3)

                    print('Imagedan segmentasyon işlemleri tamamlandı..')
                    print('main_image_detection', 'İşlem başarılı')

                elif process == 'detection':

                    print('Imagedan nesne tanıma işlemleri başlatılıyor..')

                    yolov8detector = yolov8_detector.YoloV8Detector(
                        args.job_id, data_sub_type, args.main_directory, preprocess_img_output_directory, 
                        args.model_weights, args.model_parameters, xml_file, depth_path, 
                        args.source_EPSG, args.city_id, args.main_url, args.token)

                    yolov8detector.run()

                    print('Imagedan nesne tanıma işlemleri tamamlandı..')
                    print('main_image_detection', 'İşlem başarılı')

                else:
                    raise ValueError(f"Unsupported process type: {process}")

        elif args.data_type == 'Others':

            process = DecisionUtils.decide_process_type(args.model_weights)

            if process == 'segmentation':

                print('Imagedan segmentasyon işlemleri başlatılıyor..')
                segmentation = ImageSegmentation(input_folder=args.main_directory, pass_postprocess=True, 
                                                 output_folder=args.main_directory, model_path=args.model_weights,
                                                 job_id=args.job_id,city_id=args.city_id, main_url=args.main_url, token=args.token,
                                                 source_EPSG=args.source_EPSG,
                                                 save_img=True, save_txt=True, save_crops=True)
                segmentation.load_model()
                segmentation.process_images(padding=20, mask_alpha=0.3)

                print('Imagedan segmentasyon işlemleri tamamlandı..')
                print('main_image_detection', 'İşlem başarılı')

            elif process == 'detection':

                data_type = "Others"
                xml_file = None 
                depth_path = None

                print('Imagedan nesne tanıma işlemleri başlatılıyor..')

                yolov8detector = yolov8_detector.YoloV8Detector(job_id=args.job_id,datatype= data_type,main_directory= args.main_directory, input_directory= args.main_directory, 
                    model_weights= args.model_weights,model_parameters= args.model_parameters,xml_data= xml_file, depth_main_path= depth_path, 
                    source_EPSG= args.source_EPSG,city= args.city_id,main_url= args.main_url, token=args.token)
                yolov8detector.run()

                print('Imagedan nesne tanıma işlemleri tamamlandı..')
                print('İşlem başarılı')

    except Exception as e:
        tb = traceback.format_exc()
        print('Hata')
        raise e
