import json
import os
import pandas as pd
import random
import numpy as np
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel


# Set your Google Cloud project ID
PROJECT_ID = 'psde'

features_fname = "../jsons/bach_features.json"
labels_fname = "../full_labeled_music.csv"
out_fname = "predicted_labels_gemini_music.csv"

# Load your JSON file containing music features
def load_features():
    with open(features_fname, 'r') as f:
        return json.load(f)
    
def get_labels():
    return pd.read_csv(labels_fname)

def make_few_shot_prompt_gemini(labels,predict_fname,k):
    music_features = load_features()

    # Sample k images with their ground truth labels
    fnames = list(labels['fname'])
    sample_fnames = [predict_fname]
    while predict_fname in sample_fnames:
        sample_indices = random.sample(range(len(fnames)), k)
        sample_fnames = [fnames[i] for i in sample_indices]
    standard_prompt = "You are an expert in music. Your job is to classify a chord progression as " + \
        "something I would like or dislike. You will be given the chord progression, along with some examples "+ \
        "of previous chord progressiosn that I have either liked or disliked"+\
        "one line consisting of one of the following labels: \nlike\ndislike. "+\
        "Here are a few example images:\n"
    overall_prompt = [standard_prompt]

    # Construct the few-shot prompt
    for i in sample_indices:
        fname = fnames[i]
        features = music_features.get(fname, "")
        overall_prompt.append(f"Chord progression: {features}\nLabel: {'like' if labels['val'][i] else 'dislike'}\n")
    overall_prompt.append("Chord progression to predict: ")
    features = music_features.get(predict_fname,"")
    overall_prompt.append(f"{features}\nLabel: ")
    return overall_prompt


# Define a function to run predictions with multimodal prompts
def run_predictions(model,labels):
    predictions = pd.DataFrame(columns = ['fname','val','k'])
    fnames = list(labels['fname'])
    # Iterate over images in the dataset
    total_toks = 0
    for k in range(1,15):
        i = 0
        for fname in fnames:
            i += 1
            if i > 500:
                break
            prompt = make_few_shot_prompt_gemini(labels,fname,k)
            toks_1k = np.ceil(sum([len(s) for s in prompt])/1000.0)
            total_toks += toks_1k
            print(k)
            print(total_toks*0.000125)
            model_response = model(prompt)
            try:
                label = model_response.candidates[0].content.parts[0].text.strip()
            except:
                print(model_response)
                continue
            print(label)
            if label == "like":
                predictions = predictions._append({'fname':fname,'val':True,'k':k},ignore_index=True)
            elif label == "dislike":
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
    print("dataset created")
    gemini_pro_vision_model = GenerativeModel("gemini-1.0-pro-vision")
    #gpt_model = lambda prompt: client.chat.completions.create(model="gpt-4-vision-preview",messages=prompt)
    gemini_model = lambda prompt: gemini_pro_vision_model.generate_content(prompt)
    predictions = run_predictions(gemini_model,labels)
    predictions.to_csv(out_fname)

if __name__ == "__main__":
    main()
