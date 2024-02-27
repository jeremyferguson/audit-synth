import json
import pandas as pd

features_fname = "extracted_features_detr_500.json"
csv_fname = "partial_labeled_sports.csv"
out_fname = "task.csv"

df = pd.read_csv(csv_fname)

data = []
with open(features_fname,'r') as f:
    features = json.load(f)
i = 0
for row in df.iterrows():
    i += 1
    if i > 10:
        break
    example_str = f'{",".join(features[row[1]["fname"]])}'
    #label_str = "sports" if row[1]['val'] else "not sports"
    label_str = ""
    data.append([example_str,label_str])

out_df = pd.DataFrame(columns=['example','label'],data=data)
out_df.to_csv(out_fname,index=False)