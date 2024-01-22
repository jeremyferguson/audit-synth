from PIL import Image
import os

def get_image_sizes(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter out only image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    max_length = 0
    # Iterate through image files and print their sizes
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        try:
            # Open the image
            with Image.open(image_path) as img:
                # Get the size of the image
                width, height = img.size
                if width > max_length or height > max_length:
                    max_length = max(height,width)
                print(f"Image: {image_file}, Size: {width} x {height}")
        except Exception as e:
            # Handle exceptions, e.g., if the file is not a valid image
            print(f"Error processing {image_file}: {e}")
    print(max_length)

# Example usage
if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to your image directory
    image_directory = '/home/jmfergie/coco_imgs'

    get_image_sizes(image_directory)
