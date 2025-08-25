import xml.etree.ElementTree as ET
import pandas as pd

def get_lat_lon_height_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    generated_at = root.find('{http://www.cyclomedia.com/}GeneratedAt').text

    recordings_data = []
    for recording in root.findall('.//{http://www.cyclomedia.com/}Recording'):
        data = {
            "ExpiredAt": recording.find('{http://www.cyclomedia.com/}ExpiredAt').text,
            "GroundLevelOffset": float(recording.find('{http://www.cyclomedia.com/}GroundLevelOffset').text),
            "Height": float(recording.find('{http://www.cyclomedia.com/}Height').text),
            "HeightPrecision": float(recording.find('{http://www.cyclomedia.com/}HeightPrecision').text),
            "ImageId": recording.find('{http://www.cyclomedia.com/}ImageId').text,
            "JpgQuality": int(recording.find('{http://www.cyclomedia.com/}JpgQuality').text),
            "Latitude": float(recording.find('{http://www.cyclomedia.com/}Latitude').text),
            "LatitudePrecision": float(recording.find('{http://www.cyclomedia.com/}LatitudePrecision').text),
            "Longitude": float(recording.find('{http://www.cyclomedia.com/}Longitude').text),
            "LongitudePrecision": float(recording.find('{http://www.cyclomedia.com/}LongitudePrecision').text),
            "Orientation": float(recording.find('{http://www.cyclomedia.com/}Orientation').text),
            "OrientationPrecision": float(recording.find('{http://www.cyclomedia.com/}OrientationPrecision').text),
            "ProductType": recording.find('{http://www.cyclomedia.com/}ProductType').text,
            "RecordedAt_DateTime": recording.find('{http://www.cyclomedia.com/}RecordedAt').find('{http://schemas.datacontract.org/2004/07/System}DateTime').text,
            "RecordedAt_OffsetMinutes": int(recording.find('{http://www.cyclomedia.com/}RecordedAt').find('{http://schemas.datacontract.org/2004/07/System}OffsetMinutes').text),
            "RecorderDirection": float(recording.find('{http://www.cyclomedia.com/}RecorderDirection').text),
            "RecorderGeneration": recording.find('{http://www.cyclomedia.com/}RecorderGeneration').text,
            "RecordingSystem": recording.find('{http://www.cyclomedia.com/}RecordingSystem').text,
            "SRID": int(recording.find('{http://www.cyclomedia.com/}SRID').text),
            "SystemCalibration": recording.find('{http://www.cyclomedia.com/}SystemCalibration').text,
            "TileSchema": int(recording.find('{http://www.cyclomedia.com/}TileSchema').text),
            "Dataset_Name": root.find('.//{http://www.cyclomedia.com/}Name').text,
            "GeneratedAt": generated_at
        }
        recordings_data.append(data)

    df = pd.DataFrame(recordings_data)
    return df

