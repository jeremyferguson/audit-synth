import json
import os
import pandas as pd
import random
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel



# Set your Google Cloud project ID
PROJECT_ID = 'psde'

# Set your Google Cloud Storage bucket name
BUCKET_NAME = 'coco_images_psde'

features_fname = "../extracted_features_detr.json"
labels_fname = "partial_labeled_sports.csv"
out_fname = "predicted_labels_gemini.csv"

# Load your JSON file containing image features
def load_features():
    with open(features_fname, 'r') as f:
        return json.load(f)
    
def get_labels():
    return pd.read_csv(labels_fname)

# Define a function to create an ImageDataset
def create_image_dataset(fnames):
    dataset = {
        fname: generative_models.Part.from_uri(f"gs://{BUCKET_NAME}/{fname}", mime_type="image/jpeg")
        for fname in fnames}
    return dataset

def make_few_shot_prompt_gemini(dataset,labels,predict_fname,k):
    image_features = load_features()

    # Sample k images with their ground truth labels
    fnames = list(labels['fname'])
    sample_fnames = [predict_fname]
    while predict_fname in sample_fnames:
        sample_indices = random.sample(range(len(fnames)), k)
        sample_fnames = [fnames[i] for i in sample_indices]
    standard_prompt = "You are an expert in sports. Your job is to classify an image as either " + \
        "sports-related or not sports-related. You will be given the image, along with an extracted "+ \
        "list of features that a computer vision model has found in the image. Respond with exactly "+\
        "one line consisting of one of the following labels: \nsports\nnot sports. "+\
            f"{'Here are a few example images:' if k > 0 else ''}\n"
    overall_prompt = [standard_prompt]

    # Construct the few-shot prompt
    for i in sample_indices:
        fname = fnames[i]
        features = image_features.get(fname, "")
        overall_prompt.append(dataset[fname])
        overall_prompt.append(f"Extracted features: {features}\nLabel: {'sports' if labels['val'][i] else 'not sports'}\n")
    overall_prompt.append("Image to predict: ")
    overall_prompt.append(dataset[predict_fname])
    features = image_features.get(predict_fname,"")
    overall_prompt.append(f"Extracted features: {features}\nLabel: ")
    return overall_prompt

def make_few_shot_prompt_gpt(dataset,labels,predict_fname,k):
    image_features = load_features()

    # Sample k images with their ground truth labels
    fnames = list(labels['fname'])
    sample_fnames = [predict_fname]
    while predict_fname in sample_fnames:
        sample_indices = random.sample(range(len(fnames)), k)
        sample_fnames = [fnames[i] for i in sample_indices]
    standard_prompt = "You are an expert in sports. Your job is to classify an image as either " + \
        "sports-related or not sports-related. You will be given the image, along with an extracted "+ \
        "list of features that a computer vision model has found in the image. Respond with exactly "+\
        "one line consisting of one of the following labels: \nsports\nnot sports. "+\
            f"{'Here are a few example images:' if k > 0 else ''}\n"
    overall_prompt = [{"type": "text", "text": standard_prompt}]
    # Construct the few-shot prompt
    
    print(dataset[predict_fname].to_dict())
    for i in sample_indices:
        fname = fnames[i]
        features = image_features.get(fname, "")
        overall_prompt.append({
            "type": "image_url",
            "image_url": {
                "url": dataset[fname].to_dict()['file_data']['file_uri'],
                "detail": "low"}})
        overall_prompt.append({"type": "text", "text": 
                               f"Extracted features: {features}\nLabel: {'sports' if labels['val'][i] else 'not sports'}\n"})
    overall_prompt.append({"type": "text", "text": "Image to predict: "})
    overall_prompt.append({"type": "image_url",
            "image_url": {
                "url": dataset[predict_fname].to_dict()['file_data']['file_uri'],
                "detail": "low"}})
    features = image_features.get(predict_fname,"")
    overall_prompt.append({"type":"text", "text":f"Extracted features: {features}\nLabel: "})
    return {
        "role": "user",
        "content": overall_prompt
    }

# Define a function to run predictions with multimodal prompts
def run_predictions(model,dataset,labels):
    predictions = pd.DataFrame(columns = ['fname','val','k'])
    # Iterate over images in the dataset
    for k in [0,1,5,10,15]:
        i = 0
        for fname in dataset:
            i += 1
            if i > 500:
                break
            prompt = make_few_shot_prompt_gemini(dataset,labels,fname,k)
            print(prompt)
            model_response = model(prompt)
            label = model_response.candidates[0].content.parts[0].text.strip()
            if label == "sports":
                predictions = predictions._append({'fname':fname,'val':True,'k':k},ignore_index=True)
            elif label == "not sports":
                predictions = predictions._append({"fname":fname,"val": False,'k':k},ignore_index=True)
            else:
                print("Invalid response: ",model_response)
        predictions.to_csv(out_fname)
    return predictions

# Main function
def main():
    labels = get_labels()
    print("labels created")
    fnames = list(labels['fname'])
    dataset = create_image_dataset(fnames)
    print("dataset created")
    gemini_pro_vision_model = GenerativeModel("gemini-1.0-pro-vision")
    #gpt_model = lambda prompt: client.chat.completions.create(model="gpt-4-vision-preview",messages=prompt)
    gemini_model = lambda prompt: gemini_pro_vision_model.generate_content(prompt)
    predictions = run_predictions(gemini_model,dataset,labels)
    predictions.to_csv(out_fname)

if __name__ == "__main__":
    main()
