import os
from tkinter import Tk, Label
from PIL import Image, ImageTk

class ImageSorter:
    def __init__(self, root):
        self.root = root
        self.image_dir = "path_to_your_image_directory"
        self.yes_dir = "path_to_your_yes_directory"
        self.no_dir = "path_to_your_no_directory"
        self.current_image_idx = 0
        self.images = self.get_image_files()
        self.total_images = len(self.images)
        
        self.label = Label(root)
        self.label.pack()
        
        self.status_label = Label(root, text="")
        self.status_label.pack()

        self.display_image()
        self.update_status()

        self.root.bind('y', self.move_to_yes)
        self.root.bind('n', self.move_to_no)

    def get_image_files(self):
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        return image_files

    def display_image(self):
        if self.current_image_idx < len(self.images):
            image_path = os.path.join(self.image_dir, self.images[self.current_image_idx])
            image = Image.open(image_path)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.label.config(image=photo)
            self.label.image = photo
        else:
            self.label.config(text="No more images to display")

    def move_to_yes(self, event=None):
        if self.current_image_idx < len(self.images):
            current_image = self.images[self.current_image_idx]
            source_path = os.path.join(self.image_dir, current_image)
            destination_path = os.path.join(self.yes_dir, current_image)
            os.rename(source_path, destination_path)
            self.current_image_idx += 1
            self.display_image()
            self.update_status()

    def move_to_no(self, event=None):
        if self.current_image_idx < len(self.images):
            current_image = self.images[self.current_image_idx]
            source_path = os.path.join(self.image_dir, current_image)
            destination_path = os.path.join(self.no_dir, current_image)
            os.rename(source_path, destination_path)
            self.current_image_idx += 1
            self.display_image()
            self.update_status()

    def update_status(self):
        self.status_label.config(text=f"Image {self.current_image_idx+1} out of {self.total_images}")

if __name__ == "__main__":
    root = Tk()
    root.title("Image Sorter")
    app = ImageSorter(root)
    root.mainloop()
