# initialize_labels.py

import os
import piexif
from PIL import Image
import json

STATIC_IMAGES_DIR = 'static/images'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg'}

def initialize_labels():
    images = [f for f in os.listdir(STATIC_IMAGES_DIR) if allowed_file(f)]
    for filename in images:
        image_path = os.path.join(STATIC_IMAGES_DIR, filename)
        try:
            img = Image.open(image_path)
            
            # Attempt to retrieve existing EXIF data
            exif_bytes = img.info.get('exif', None)
            
            if exif_bytes:
                try:
                    exif_dict = piexif.load(exif_bytes)
                except piexif.InvalidImageDataError:
                    print(f"Invalid EXIF data for {filename}, initializing new EXIF.")
                    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            else:
                # No EXIF data present
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            
            # Initialize all labels to False
            labels = {
                "cat": False,
                "morris": False,
                "entering": False,
                "prey": False
            }
            description = json.dumps(labels)
            exif_dict['0th'][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
            exif_bytes_new = piexif.dump(exif_dict)
            img.save(image_path, "jpeg", exif=exif_bytes_new)
            print(f"Initialized labels for {filename}")
        except FileNotFoundError:
            print(f"Error initializing labels for {filename}: File not found.")
        except piexif.InvalidImageDataError as e:
            print(f"Error initializing labels for {filename}: {e}")
        except Exception as e:
            print(f"Error initializing labels for {filename}: {e}")

if __name__ == "__main__":
    initialize_labels()
