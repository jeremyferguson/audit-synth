root_dir = "/home/jmfergie/streetview_images_2/intersections"
yes_dir = "/home/jmfergie/streetview_images_2/intersections/safe"
no_dir = "/home/jmfergie/streetview_images_2/intersections/unsafe"
out_fname = "full_labeled.csv"
import os
import pandas as pd

df = pd.DataFrame()
df['fname'] = []
df['val'] = []

for fname in os.listdir(yes_dir):
        df.loc[len(df.index)] = [fname,True]
        source_path = os.path.join(yes_dir,fname)
        destination_path = os.path.join(root_dir, fname)
        os.rename(source_path, destination_path)

for fname in os.listdir(no_dir):
        df.loc[len(df.index)] = [fname,False]
        source_path = os.path.join(no_dir,fname)
        destination_path = os.path.join(root_dir, fname)
        os.rename(source_path, destination_path)

df.to_csv(out_fname)

