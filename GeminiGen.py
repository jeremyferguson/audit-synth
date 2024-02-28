import json
import numpy as np

import os
import random
from collections import Counter
import vertexai
import http.client
import typing
import urllib.request
from vertexai.generative_models import GenerativeModel, Image

features_fname = "extracted_features_gemini_500_2.json"
img_dir = "/home/jmfergie/coco_imgs"

# Set your Google Cloud project ID
PROJECT_ID = 'psde'

# Set your Google Cloud Storage bucket name
BUCKET_NAME = 'coco_images_psde'

class Extractor:

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith(('.JPG', '.jpg', '.jpeg', '.png', '.gif'))]
        self.model = GenerativeModel("gemini-1.0-pro-vision-001")
    
    def genPrompt(self,image):
        responses = self.model.generate_content(
            ["Please respond with all the objects you recognize in this image. Output them in the form of a Python list, \
              e.g. [\"bagel\", \"lizard\"].", image, "Objects:"],
            generation_config={
                "max_output_tokens": 400,
                "temperature": 0.4,
                "top_p": 1,
                "top_k": 32
            },
            stream=True,
        )
        return responses
    
    def extract(self):
        objects = {}
        for i in range(10,15):
            fname = self.filenames[i]
            image = Image.load_from_file(f"{self.img_dir}/{fname}")
            
            responses = self.genPrompt(image)
            for response in responses:
                output = response.text
                print(output)
                if output.count('"') % 2 == 1:
                    output += '"]'
                if output[-1] != ']':
                    output += ']'
                print(eval(output))
                objects[fname] = eval(output)
        return objects


if __name__ == "__main__":
    extractor = Extractor(img_dir)
    obj_dict = extractor.extract()
    print("Extraction done")
    with open(features_fname,'w') as f:
        f.write(json.dumps(obj_dict))   
