from PIL import Image
import os

# Configuration
INPUT_FOLDER = "images/no_mask_detection"          # Folder containing original images
OUTPUT_FOLDER = "images_resized/no_mask_detection" # Folder to save resized images (will be created)
TARGET_SIZE = (400, 400)        # (width, height) in pixels

def resize_images():
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Loop through all files in the input folder
    for filename in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, filename)
        
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            continue
        
        try:
            # Open, resize, and save the image
            with Image.open(input_path) as img:
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                output_path = os.path.join(OUTPUT_FOLDER, filename)
                img.save(output_path)
                print(f"Resized: {filename} -> {TARGET_SIZE}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    resize_images()