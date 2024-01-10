class Document:


class Program:


class Pred:


class Exists(Pred):

class Not(Pred):

class AndPred(Pred):

class OrPred(Pred):

def freq_heuristic():
    
class App:
    def __init__(self,heuristic,features_fname,examples_fname):
        self.heuristic = heuristic
        with open(features_fname,'r') as f:

        with open(examples_fname,'r') as f:

        
    def loop(self):
        while True:
            self.features = 


features_fname = "img_features.json"
examples_fname = "img_examples.csv"

if __name__ == "__main__":
    app = App()
    app()