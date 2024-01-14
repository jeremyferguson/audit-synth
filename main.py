import json
import pandas as pd

class Document:
    def __init__(self,fname,features):
        self.features = features
        self.fname = fname


class Program:
    def __init__(self):

    def add_pred(self,pred):

    def eval(self,doc)

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

def user_input_yn(prompt,default=True):
    prompt += f"\nEnter y/n. (Default: {'y' if default else 'n' }): "
    response = input(prompt)
    while response not in ['y','n','']:
        response = input("Please enter y or n and try again: ")
    if response == 'y':
        return True
    elif response == 'n':
        return False
    else:
        return default
    
def freq_heuristic():
    
class App:
    def __init__(self,heuristic,features_fname,examples_fname):
        self.heuristic = heuristic
        with open(features_fname,'r') as f:
            self.features = json.load(f.read())
        self.user_examples = pd.read_csv(examples_fname)

        
    def loop(self):
        prog = Program()
        rejected_preds = set()
        prog_complete = user_input_yn(f"Current program is {prog}. Are you satisfied with this?",default = False)
        while not prog_complete:
            new_preds = self.heuristic(self.prog,self.features)
            new_preds_str = '\n'.join([str(pred) for pred in new_preds])
            add_preds = user_input_yn(f"New predicates generated: {new_preds_str}\n. Would you like to add all of these to your program?")
            if add_preds:
                for pred in new_preds:
                    prog.add_pred(pred)
            else:
                add_any = user_input_yn(f"Would you like to add any of these to your program?")
                if add_any:
                    for pred in new_preds:
                        add_pred = user_input_yn(f"Would you like to add this predicate to your program: {pred}")
                        if add_pred:
                            prog.add_pred(pred)
                        else:
                            rejected_preds.add(pred)
                else:
                    for pred in new_preds:
                        rejected_preds.add(pred)
                    
            

features_fname = "img_features.json"
examples_fname = "img_examples.csv"

if __name__ == "__main__":
    app = App()
    app.loop()