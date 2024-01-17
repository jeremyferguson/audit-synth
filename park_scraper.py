import google_streetview.api
import google_streetview.helpers
import pandas as pd
import os

img_dir = "/home/jmfergie/park_streetview"
csv_fname = "/home/jmfergie/SF_parks_top1k.csv"
api_key = os.environ['STREETVIEW_API_KEY']

df = pd.read_csv(csv_fname)
params = []
i = 0
for _, row in df.iterrows():
    i += 1
    #print(row)
    location = f"{row['NEAR_Y']},{row['NEAR_X']}"
    heading = str(int(row['SF_parks_points_Perp_angle']))
    print(location)
    params.append({
        'size': '400x400', # max 640x640 pixels
        'location': location,
        'heading': heading,
        'key': api_key
    })
results = google_streetview.api.results(params)
#print(results.links)
os.chdir(img_dir)
results.download_links('imgs')