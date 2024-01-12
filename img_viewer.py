import os
import json
from tkinter import Tk, Label
from PIL import Image, ImageTk

class ImageNavigator:
    def __init__(self, root, img_dir):
        self.root = root
        self.image_dir = img_dir
        self.current_image_idx = 0
        self.images = self.get_image_files()
        self.json_data = self.load_json_data()  # Load JSON data

        self.label = Label(root)
        self.label.pack()
        
        self.filename_label = Label(root, text="")
        self.filename_label.pack()

        self.display_image()
        self.update_filename()

        self.root.bind('<Left>', self.show_previous_image)
        self.root.bind('<Right>', self.show_next_image)

    def get_image_files(self):
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        return image_files

    def load_json_data(self):
        json_file_path = "captions.json"  # Replace with your JSON file path
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def display_image(self):
        if 0 <= self.current_image_idx < len(self.images):
            image_path = os.path.join(self.image_dir, self.images[self.current_image_idx])
            image = Image.open(image_path)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.label.config(image=photo)
            self.label.image = photo
        else:
            self.label.config(text="No more images to display")

    def update_filename(self):
        if 0 <= self.current_image_idx < len(self.images):
            filename = self.images[self.current_image_idx]
            value = self.json_data.get(filename, "No value found")
            self.filename_label.config(text=f"Filename: {filename}\nValue: {value}")
        else:
            self.filename_label.config(text="")

    def show_previous_image(self, event):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.display_image()
            self.update_filename()

    def show_next_image(self, event):
        if self.current_image_idx < len(self.images) - 1:
            self.current_image_idx += 1
            self.display_image()
            self.update_filename()

if __name__ == "__main__":
    root = Tk()
    img_dir = "/home/jmfergie/streetview-images"
    root.title("Image Navigator")
    app = ImageNavigator(root,img_dir)
    root.mainloop()