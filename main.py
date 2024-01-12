class Document:
    def __init__(self,fname,features):
        self.features = features
        self.fname = fname


class Program:


class Pred:
    def eval(self,doc):
        pass

class Exists(Pred):
    def __init__(self, feature):
        self.feature = feature

    def eval(self,doc):
        return self.feature in doc.features

class Not(Pred):
    def __init__(self,pred):
        self.pred = pred

    def eval(self,doc):
        return not self.pred.eval(doc)

class AndPred(Pred):
    def __init__(self,pred1, pred2):
        self.pred1 = pred1
        self.pred2 = pred2

    def eval(self,doc):
        return self.pred1.eval(doc) and self.pred2.eval(doc)

class OrPred(Pred):
     def __init__(self,pred1, pred2):
        self.pred1 = pred1
        self.pred2 = pred2

    def eval(self,doc):
        return self.pred1.eval(doc) or self.pred2.eval(doc)

class MultiPred(Pred):
    def __init__(self,preds,thresh):
        self.preds = preds
        self.thresh = thresh
    
    def eval(self,doc):
        score = 0
        for pred in self.preds:
            if pred.eval(doc):
                score += 1
        return score >= score.thresh

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