from autolabel import LabelingAgent, AutolabelDataset, get_data
import json
#get_data('movie_reviews')
with open('config.json','r') as f:
    config=json.load(f)
print(config)
agent = LabelingAgent(config)
ds = AutolabelDataset('task.csv', config = config)
ds = agent.run(ds)
for row in ds.df.iterrows():
    print(row[1]['label_annotation'])
