import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, DetrForObjectDetection
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
from torchvision import transforms
from collections import Counter

features_fname = "extracted_features_detr_500.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
img_dir = "/home/jmfergie/coco_imgs"

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

class ImgDataset(Dataset):
    def __init__(self, img_dir,processor,max_length=640):
        self.img_dir = img_dir
        self.processor = processor
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith(('.JPG', '.jpg', '.jpeg', '.png', '.gif'))]
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((max_length, max_length)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        input_tensor = self.preprocess(filename)
        return input_tensor

    def preprocess(self, filename):
        image = torch.Tensor(Image.open(f"{self.img_dir}/{filename}").convert("RGB"))
        #image = self.transform(image)
        return image,filename

class Extractor:
    
    def __init__(self, img_dir,processor,model,batch_size=10):
        self.img_dir = img_dir
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith(('.JPG', '.jpg', '.jpeg', '.png', '.gif'))]
        self.batch_size = batch_size
        self.model = model
        self.processor = processor
        self.dataset = ImgDataset(img_dir,processor)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        
    def extract(self):
        objects = {}
        for i in range(0,500,self.batch_size):
            if i % 100 == 0:
                print(i)
            fnames = self.filenames[i:i+self.batch_size]
            images = [Image.open(f"{self.img_dir}/{filename}").convert("RGB") for filename in fnames]
            inputs = self.processor(images=images,return_tensors="pt").to(device)
            outputs = model(**inputs)
            results = self.postprocess(images,fnames,outputs)
            for res in results:
                objects[res] = list(set(results[res]))
        '''for batch,fnames in self.dataloader:
            inputs = self.processor(batch)
            inputs['pixel_values'] = torch.Tensor(np.array(inputs['pixel_values'])).to(device)
            inputs['pixel_mask'] = torch.Tensor(np.array(inputs['pixel_mask'])).to(device)
            outputs = model(**inputs)
            results = self.postprocess(inputs,fnames,outputs)
            print(results)'''
        return objects

    def postprocess(self,inputs,fnames,outputs):
        target_sizes = torch.tensor([image.size[::-1] for image in inputs])
        results = self.processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
        features = {}
        for fname, result in zip(fnames,results):
            features[fname] = [self.model.config.id2label[label.item()] for label in result['labels']]
        return features


if __name__ == "__main__":
    extractor = Extractor(img_dir,image_processor,model,batch_size=2)
    obj_dict = extractor.extract()
    print("Extraction done")
    with open(features_fname,'w') as f:
        f.write(json.dumps(obj_dict))   
