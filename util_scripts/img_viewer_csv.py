import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import pandas as pd
import os
import json

class ImageViewerApp:
    def __init__(self, root, csv_file, root_directory, json_file,title,image_list=None):
        self.root = root
        self.root.title(title)
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("q",root.destroy)

        self.image_index = 0
        if image_list:
            self.image_list = [os.path.join(root_directory, fname) for fname in image_list]
        else:
            self.image_list = self.load_images_from_csv(csv_file, root_directory)
        self.json_data = self.load_json_data(json_file)

        self.display_image()

    def load_images_from_csv(self, csv_file, root_directory):
        df = pd.read_csv(csv_file)
        image_list = [os.path.join(root_directory, fname) for fname in df['fname']]
        return image_list

    def load_json_data(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        return data

    def display_image(self):
        image_path = self.image_list[self.image_index]
        img = Image.open(image_path)
        img = img.resize((400, 400))
        photo = ImageTk.PhotoImage(img)

        if hasattr(self, 'label_image'):
            self.label_image.configure(image=photo)
            self.label_image.image = photo
        else:
            self.label_image = tk.Label(self.root, image=photo)
            self.label_image.pack()

        # Display current filename below the image
        filename_label_text = os.path.basename(image_path)
        if hasattr(self, 'label_filename'):
            self.label_filename.configure(text=filename_label_text)
        else:
            self.label_filename = tk.Label(self.root, text=filename_label_text)
            self.label_filename.pack()

        # Display data associated with the filename
        filename_key = filename_label_text
        if filename_key in self.json_data:
            data_label_text = f"Data: {self.json_data[filename_key]}"
            if hasattr(self, 'label_data'):
                self.label_data.configure(text=data_label_text)
            else:
                self.label_data = tk.Label(self.root, text=data_label_text)
                self.label_data.pack()

    def next_image(self, event):
        self.image_index = (self.image_index + 1) % len(self.image_list)
        self.display_image()

    def prev_image(self, event):
        self.image_index = (self.image_index - 1) % len(self.image_list)
        self.display_image()

def launch_app(csv_file,root_directory, json_file,title,image_list=None):
    root = tk.Tk()
    app = ImageViewerApp(root, csv_file, root_directory, json_file,title,image_list)
    root.mainloop()

if __name__ == "__main__":
    csv_file = "partial_labeled_sports.csv" 
    root_directory = "/home/jmfergie/coco_imgs"  
    json_file = "extracted_features_gemini_500_5.json"
    title = "Image viewer app"

    launch_app(csv_file, root_directory, json_file, title)
