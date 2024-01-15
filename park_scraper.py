import google_streetview.api
import pandas as pd
import os

csv_fname = "/home/jeremy/Raleighparkpoints_selected_Table.csv"
api_key = os.environ['STREETVIEW_API_KEY']

df = pd.read_csv(csv_fname)
for row in df:
    location = f"{row['LAT']},{row['LON']}"
    heading = str(row['perp_angle'])
    params = [{
        'size': '400x400', # max 640x640 pixels
        'location': location,
        'heading': heading
        'key': api_key
    }]
    results = google_streetview.api.results(params)
    results.download_links('imgs')