import os
from PIL import Image
from datetime import datetime

def process_image(image_path, resized_output_folder, cropped_output_folder):
    # Open the image
    img = Image.open(image_path)
    width, height = img.size

    # Define the crop size (square: 384x384)
    crop_size = 384

    # Calculate the coordinates for a center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop the center square from the image
    cropped_img = img.crop((left, top, right, bottom))

    # Resize the cropped image to 96x96 using LANCZOS resampling
    resized_img = cropped_img.resize((96, 96), Image.Resampling.LANCZOS)

    # Save the manipulated image to the output folder with the same filename
    base_name = os.path.basename(image_path)
    resized_output_path = os.path.join(resized_output_folder, base_name)
    cropped_output_path = os.path.join(cropped_output_folder, base_name)
    resized_img.save(resized_output_path)
    cropped_img.save(cropped_output_path)
    print(f"Processed: {base_name}")

def main():
    input_folder = "images"
    
    # Create a new output folder with current date and time in its name
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    resized_output_folder = f"resized_{now}"
    cropped_output_folder = f"cropped_{now}"
    os.makedirs(resized_output_folder, exist_ok=True)
    os.makedirs(cropped_output_folder, exist_ok=True)
    
    # Loop over all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, resized_output_folder, cropped_output_folder)

if __name__ == "__main__":
    main()
