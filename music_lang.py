import os

class Chord:
    def __init__(self,name):
        self.name = name

    def eval(self,doc):
        return self.name in doc.features
    
class ChordSeq:
    def __init__(self,seq):
        self.seq = seq

    def eval(self,doc):
        for i in range(0,len(doc.features)-len(self.seq)):
            if self.seq == doc.features[i:i+len(self.seq)]:
                return True
        return False

def parse_music_pred():
    pass

def music_freq_heuristic():
    pass

class Document:
    def __init__(self,fname,features):
        self.features = list(features)
        self.fname = fname

