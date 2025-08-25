import math
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

def get_records_within_buffer(df, target_lat, target_lon, radius_meters=10):
    nearby_records = []
    
    for index, row in df.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        
        distance = haversine(target_lat, target_lon, lat, lon)
        
        if distance <= radius_meters:
            nearby_records.append(row)
    
    return pd.DataFrame(nearby_records)

def check_detection_fp_inthreshold(xml_data, target_lat, target_lon, target_z, min_height, max_height, ShootingPoint_height):

    detection_fp_inthreshold_result = {
        'result': False,
        'nearby_height_min': None,
        'nearby_height_max': None,
        'average_ground_offset': None,
        'ground_level_min_ilk': None,
        'ground_level_max_ilk': None,
        'ground_level_min_son': None,
        'ground_level_max_son': None,
        'target_z': target_z,
        'shooting_point_height': ShootingPoint_height
    }

    nearby_data = get_records_within_buffer(xml_data, target_lat, target_lon, 30)
    
    if not nearby_data.empty:

        nearby_height_min = nearby_data['Height'].min()
        nearby_height_max = nearby_data['Height'].max()
        detection_fp_inthreshold_result['nearby_height_min'] = nearby_height_min
        detection_fp_inthreshold_result['nearby_height_max'] = nearby_height_max
        average_ground_offset = nearby_data['GroundLevelOffset'].mean()
        detection_fp_inthreshold_result['average_ground_offset'] = average_ground_offset
        
        ground_level_min = nearby_height_min - average_ground_offset
        ground_level_max = nearby_height_max - average_ground_offset

        detection_fp_inthreshold_result['ground_level_min_ilk'] = ground_level_min
        detection_fp_inthreshold_result['ground_level_max_ilk'] = ground_level_max
        
        ground_level_min = ground_level_min + min_height
        ground_level_max = ground_level_max + max_height
        
        detection_fp_inthreshold_result['ground_level_min_son'] = ground_level_min
        detection_fp_inthreshold_result['ground_level_max_son'] = ground_level_max
        
        if target_z >= ground_level_min and target_z <= ground_level_max:
            detection_fp_inthreshold_result['result'] = False
        else:
            detection_fp_inthreshold_result['result'] = True
    else:
        detection_fp_inthreshold_result['result'] = False
    
    return detection_fp_inthreshold_result
