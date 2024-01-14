import json
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, AutoProcessor, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import os
import random

out_fname = "captions.json"
device = torch.device("cuda")
img_dir = "/home/jmfergie/streetview-images"
from transformers import BlipProcessor, BlipForConditionalGeneration

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

class Captioner:

    def __init__(self, img_dir,processor,model,batch_size=500,k=10):
        img_fnames = os.listdir(img_dir)
        self.fname_map = {i:fname for i,fname in enumerate(img_fnames)}
        self.img_fnames = img_fnames
        self.images = [Image.open(f"{img_dir}/{fname}") for fname in img_fnames[:batch_size]]
        self.k = k
        self.n = len(img_fnames)
        self.batch_size = batch_size
        self.gen_context_imgs()
        self.model = model
        self.processor = processor

    def gen_context_imgs(self):
        
        n = len(self.images)
        self.img_contexts = np.zeros((self.n,self.k),np.int32)
        img_indices = np.arange(n)
        img_weights = np.ones(n)
        for i in range(n):
            img_weights[i-1] = 1
            img_weights[i] = 0
            context_indices = random.choices(img_indices,weights=img_weights,k=self.k)
            self.img_contexts[i,:] = context_indices
    
    def literal_listener(self,captions, images):
        """
        Args:
            captions: list of N captions
            images: list of N images
        Returns:
            a NxN array of normalized conditional probabilities,
            where A[i][j] = p(image j | caption i)
        """
        n = len(images)
        inputs = processor(text=captions, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        scores = outputs.logits_per_image
        listener_probs = F.softmax(scores,dim=1).cpu().detach().numpy()
        assert np.allclose(listener_probs.sum(1), 1)
        return listener_probs


    def pragmatic_speaker(self,captions, images):
        """
        Args:
            captions: list of N captions
            images: list of N images
        Returns:
            a NxN array of normalized conditional probabilities,
            where A[i][j] = p(caption i | image j)
        """
        listener_probs = self.literal_listener(captions, images)
        caption_sums = np.sum(listener_probs,0)
        speaker_probs = listener_probs/caption_sums
        assert np.allclose(speaker_probs.sum(0), 1)
        return speaker_probs


    def pragmatic_listener(self,captions, images):
        """
        Args:
            captions: list of N captions
            images: list of N images
        Returns:
            a NxN array of normalized conditional probabilities,
            where A[i][j] = p(image j | caption i)
        """
        speaker_probs = self.pragmatic_speaker(captions, images)
        speaker_sums = np.sum(speaker_probs,1)
        listener_probs = (speaker_probs.T/speaker_sums).T
        assert np.allclose(listener_probs.sum(1), 1)
        return listener_probs

    def generate_candidate_captions(self,image):
        """Generate k candidate captions for an image."""
        inputs = self.processor(image,return_tensors="pt").to(device)
        
        out = self.model.generate(**inputs,max_new_tokens=50,do_sample=True,num_return_sequences=self.k+1,temperature=0.7)
        candidates = self.processor.batch_decode(out,skip_special_tokens=True)
        assert isinstance(candidates, list) and len(candidates) == self.k+1 and isinstance(candidates[0], str)
        return candidates
        
    def pragmatic_captioner(self,img_index):
        """Generate a caption for the image using a set of context images."""
        image = self.images[img_index]
        candidate_captions = []
        context_images = [self.images[i] for i in self.img_contexts[img_index]]
        images = [image] + context_images
        candidate_captions = self.generate_candidate_captions(image)
        caption_probs = self.pragmatic_listener(candidate_captions,images)
        top1 = np.argmax(caption_probs,1)[0]
        # Rank captions with the pragmatic_speaker and return the best one
        pragmatic_caption = candidate_captions[top1]
        return pragmatic_caption

    def gen_captions(self,img_indices=None):
        captions = {}
        if not img_indices:
            n = len(self.images)
            img_indices = np.arange(n)
        num = 0
        for i in img_indices:
            if num % 10 == 0:
                print(num)
            num += 1
            captions[self.fname_map[i]] = self.pragmatic_captioner(i)
        return captions

if __name__ == "__main__":
    cap = Captioner(img_dir,cap_processor, cap_model)
    captions = cap.gen_captions()
    with open(out_fname, "w") as f:
        f.write(json.dumps(captions))        
