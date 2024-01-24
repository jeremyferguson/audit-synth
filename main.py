import json
import pandas as pd
import yaml
from queue import PriorityQueue

class Document:
    def __init__(self,fname,features):
        self.features = set(features)
        self.fname = fname


class Program:
    def __init__(self,low_thresh,high_thresh):
        self.preds = []
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    def add_pred(self,pred):
        self.preds.append(pred)

    def eval(self,doc):
        score = 0
        if self.preds:
            for pred in self.preds:
                if pred.eval(doc):
                    score += 1
            fraction = score / len(self.preds)
            if fraction >= self.high_thresh:
                return 2
            elif fraction >= self.low_thresh:
                return 1
            else:
                return 0
        else:
            return 0

    def __repr__(self):
        if not self.preds:
            return "Empty"
        return "\n".join([str(pred) for pred in self.preds])
    
    def __len__(self):
        return len(self.preds)

    def __contains__(self,pred):
        return pred in self.preds

class Pred:
    def eval(self,doc):
        pass

class Exists(Pred):
    def __init__(self, feature):
        self.feature = feature

    def eval(self,doc):
        return self.feature in doc.features

    def __repr__(self):
        return f"Exists({self.feature})"

    def __eq__(self,other):
        return isinstance(other,Exists) and self.feature == other.feature
    
    def __hash__(self):
        return hash(("Exists",self.feature))


class Not(Pred):
    def __init__(self,pred):
        self.pred = pred

    def eval(self,doc):
        return not self.pred.eval(doc)

    def __repr__(self):
        return f"Not({self.pred})"

    def __eq__(self,other):
        return isinstance(other,Not) and self.pred == other.pred

    def __hash__(self):
        return hash(("Not",self.pred))

class AndPred(Pred):
    def __init__(self,pred1, pred2):
        self.pred1 = pred1
        self.pred2 = pred2

    def eval(self,doc):
        return self.pred1.eval(doc) and self.pred2.eval(doc)

    def __repr__(self):
        return f"And({self.pred1},{self.pred2})"

    def __eq__(self,other):
        return isinstance(other,AndPred) and ((self.pred1 == other.pred1 and self.pred2 == other.pred2) or 
        (self.pred2 == other.pred1 and self.pred1 == other.pred2))

    def __hash__(self):
        return hash(("And",self.pred1,self.pred2))
    

class OrPred(Pred):
    def __init__(self,pred1, pred2):
        self.pred1 = pred1
        self.pred2 = pred2
    
    def eval(self,doc):
        return self.pred1.eval(doc) or self.pred2.eval(doc)

    def __repr__(self):
        return f"Or({self.pred1},{self.pred2})"

    def __eq__(self,other):
        return isinstance(other,OrPred) and ((self.pred1 == other.pred1 and self.pred2 == other.pred2) or 
        (self.pred2 == other.pred1 and self.pred1 == other.pred2))

    def __hash__(self):
        return hash(("Or",self.pred1,self.pred2))
        
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
    
def predGen(features,depth):
    def ZeroDepth():
        for feature in features:
            yield Exists(feature)
    if depth == 0:
        for pred in ZeroDepth():
            yield pred
    else:
        for i, pred1 in enumerate(ZeroDepth()):
            yield pred1
            for j, pred2 in enumerate(ZeroDepth()):
                if i < j:
                    yield AndPred(pred1,pred2)
                    yield OrPred(pred1, pred2)
    
def freq_heuristic(program, docs, features_set,examples, rejected_preds, config):
    topk = PriorityQueue(config.predicates_per_round)
    i = 0
    for pred in predGen(features_set,config.depth):
        i += 1
        if i % 100 == 0:
            print(i)
        if pred in program or pred in rejected_preds or Not(pred) in program or Not(pred) in rejected_preds:
            continue
        user_good = 0
        nonuser_good = 0
        user_bad = 0
        nonuser_bad = 0
        ex_fname = examples['fname'][0]
        for fname in docs:
            doc = docs[fname]
            if pred.eval(doc):
                if fname in examples['fname'].values:
                    user_good += 1
                else:
                    nonuser_good += 1
            else:
                if fname in examples['fname'].values:
                    user_bad += 1
                else:
                    nonuser_bad += 1
        user_approved = user_good / (user_good + user_bad)
        full_docs = (user_good + nonuser_good) / (user_good + user_bad + nonuser_good + nonuser_bad)
        score = user_approved - full_docs
        item_i = (abs(score),i,pred,score)
        if not topk.full():
            topk.put(item_i)
        else:
            item_k = topk.get()
            to_push = max(item_i,item_k,key=lambda item:item[0])
            topk.put(to_push)
    preds = []
    while not topk.empty():
        item_i = topk.get()
        print(item_i)
        if item_i[3] < 0:
            preds.append(Not(item_i[2]))
        else:
            preds.append(item_i[2])
    return preds



class AppConfig:
    def __init__(self, low_threshold=0.2, high_threshold=0.8, features_fname=None, examples_fname=None,
                 num_feature_selection_rounds=5, features_combine_count=5,depth=0,
                 predicates_per_round=5, use_mutual_information=False,manual=False):
        # Validate thresholds
        assert 0 <= low_threshold <= 1, "Low threshold must be between 0 and 1."
        assert 0 <= high_threshold <= 1, "High threshold must be between 0 and 1."
        assert low_threshold < high_threshold, "Low threshold must be less than high threshold."

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.features_fname = features_fname or "features.json"
        self.examples_fname = examples_fname or "user_examples.csv"
        self.num_feature_selection_rounds = num_feature_selection_rounds
        self.features_combine_count = features_combine_count
        self.predicates_per_round = predicates_per_round
        self.use_mutual_information = use_mutual_information
        self.manual = manual
        self.depth = depth

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)



    
class App:
    def __init__(self,heuristic,config):
        self.heuristic = heuristic
        self.config = config
        self.docs = {}
        with open(config.features_fname,'r') as f:
            json_dict = json.loads(f.read())
            for doc in json_dict:
                self.docs[doc] = Document(doc,json_dict[doc])
        self.all_features = set()
        for fname in self.docs:
            for feature in self.docs[fname].features:
                self.all_features.add(feature)

        self.user_examples = pd.read_csv(config.examples_fname)
        self.prog = Program(config.low_threshold,config.high_threshold)

        
    def run(self):
        if self.config.manual:
            self.runManual()
        else:
            self.runAutomatic()
        return self.prog

    def runAutomatic(self):
        for _ in range(self.config.num_feature_selection_rounds):
            new_preds = self.heuristic(self.prog,self.docs,self.all_features,self.user_examples,set(),self.config)
            for pred in new_preds:
                self.prog.add_pred(pred)

    def runManual(self):
        rejected_preds = set()
        prog_complete = user_input_yn(f"Current program is: \n{self.prog}\nAre you satisfied with this?",default = False)
        while not prog_complete:
            new_preds = self.heuristic(self.prog,self.docs,self.all_features,self.user_examples,rejected_preds,self.config)
            new_preds_str = '\n'.join([str(pred) for pred in new_preds])
            add_preds = user_input_yn(f"New predicates generated: \n{new_preds_str}\n. Would you like to add all of these to your program?")
            if add_preds:
                for pred in new_preds:
                    self.prog.add_pred(pred)
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
            prog_complete = user_input_yn(f"Current program is {self.prog}. Are you satisfied with this?",default = False)
                    
            
yaml_filename = "config.yml"

if __name__ == "__main__":
    config = AppConfig.from_yaml(yaml_filename)
    app = App(freq_heuristic,config)
    prog = app.run()
    print("Final program: ",prog)