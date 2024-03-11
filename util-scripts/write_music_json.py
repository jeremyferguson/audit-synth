import json, os
source_dir = "/home/jmfergie/music/output"
out_fname = "bach_features.json"
def parse_all_docs():
    docs = {}
    for fname in os.listdir(source_dir):
        with open(os.path.join(source_dir,fname),'r') as f:
            text = f.read()
        chords = text.split(',')
        docs[fname] = chords
    return docs

docs = parse_all_docs()
with open(out_fname,'w') as f:
    json.dump(docs,f)