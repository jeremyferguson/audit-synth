import os
from tkinter import Tk, Label
from PIL import Image, ImageTk

class ImageSorter:
    def __init__(self, root):
        self.root = root
        self.image_dir = "/home/jmfergie/coco_imgs"
        self.yes_dir = "/home/jmfergie/coco_imgs/sports"
        self.no_dir = "/home/jmfergie/coco_imgs/not_sports"
        self.current_image_idx = 0
        self.images = self.get_image_files()
        
        self.label = Label(root)
        self.label.pack()
        self.display_image()

        self.root.bind('y', self.move_to_yes)
        self.root.bind('n', self.move_to_no)

    def get_image_files(self):
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.JPG', '.jpg', '.jpeg', '.png', '.gif'))]
        return image_files

    def display_image(self):
        if self.current_image_idx < len(self.images):
            image_path = os.path.join(self.image_dir, self.images[self.current_image_idx])
            image = Image.open(image_path)
            image = image.resize((300, 300), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.label.config(image=photo)
            self.label.image = photo
        else:
            self.label.config(text="No more images to display")

    def move_to_yes(self, event=None):
        # Move the current image to the "yes" directory
        if self.current_image_idx < len(self.images):
            current_image = self.images[self.current_image_idx]
            source_path = os.path.join(self.image_dir, current_image)
            destination_path = os.path.join(self.yes_dir, current_image)
            os.rename(source_path, destination_path)
            self.current_image_idx += 1
            self.display_image()

    def move_to_no(self, event=None):
        # Move the current image to the "no" directory
        if self.current_image_idx < len(self.images):
            current_image = self.images[self.current_image_idx]
            source_path = os.path.join(self.image_dir, current_image)
            destination_path = os.path.join(self.no_dir, current_image)
            os.rename(source_path, destination_path)
            self.current_image_idx += 1
            self.display_image()

if __name__ == "__main__":
    root = Tk()
    root.title("Image Sorter")
    app = ImageSorter(root)
    root.mainloop()
