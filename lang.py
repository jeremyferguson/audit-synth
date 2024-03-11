from pyparsing import *
from queue import PriorityQueue
import numpy as np
from util_scripts.write_midis import create_midi
import os
import pygame
import time


pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)

class Lang:
    def __init__(self):
        self.preds = {}
        
    class Program:
        def __init__(self,low_thresh,high_thresh,preds=None):
            if not preds:
                self.preds = []
            else:
                self.preds = preds
            self.low_thresh = low_thresh
            self.high_thresh = high_thresh

        def add_pred(self,pred):
            self.preds.append(pred)

        def eval(self,doc,use_bins=True):
            score = 0
            if self.preds:
                for pred in self.preds:
                    if pred.eval(doc):
                        score += 1
                fraction = score / len(self.preds)
                if use_bins:
                    if fraction >= self.high_thresh:
                        return 2
                    elif fraction >= self.low_thresh:
                        return 1
                    else:
                        return 0
                else:
                    return score, len(self.preds)
            else:
                return 0, 0

        def __repr__(self):
            if not self.preds:
                return "Empty"
            return "\n".join([str(pred) for pred in self.preds])
        
        def __len__(self):
            return len(self.preds)

        def __contains__(self,pred):
            return pred in self.preds
        
    def mutual_info(self,pred1,pred2,docs):
        A_count = 0
        B_count = 0
        AB_count = 0
        if pred2 == None:
            for fname in docs:
                doc = docs[fname]
                if pred1.eval(doc):
                    A_count += 1
            return A_count / len(docs)
        else:
            for fname in docs:
                doc = docs[fname]
                in_A = pred1.eval(doc)
                in_B = pred2.eval(doc)
                if in_A and in_B:
                    AB_count += 1
                if in_A:
                    A_count += 1
                if in_B:
                    B_count += 1
            total = len(docs)
            p_AB = AB_count / total
            p_A = A_count / total
            p_B = B_count / total
            if (p_AB) == 0:
                return 0
            mi = p_AB * np.log(p_AB / (p_A * p_B))
            return mi
        
    def filter_topk(self,k,topk,docs):
        combined_pred = None
        top_preds = []
        all_preds = set()
        i = 0
        while not topk.empty():
            item_i = topk.get()
            i += 1
            if item_i[3] < 0:
                all_preds.add(self.Not(item_i[2]))
            else:
                all_preds.add(item_i[2])
        for _ in range(k):
            mis = {}
            for pred in all_preds:
                mi = self.mutual_info(pred,combined_pred,docs)
                mis[pred] = mi
            if not mis:
                break
            if not combined_pred:
                max_pred = max(mis,key=lambda pred:mis[pred])
            else:
                max_pred = min(mis,key=lambda pred:mis[pred])
            all_preds.remove(max_pred)
            top_preds.append(max_pred)
            if combined_pred:
                combined_pred = self.OrPred(max_pred,combined_pred)
            else:
                combined_pred = max_pred
        return top_preds
    
    def freq_heuristic(self,program, docs, features_set,examples, rejected_preds, config):
        top_docs = set()
        bottom_docs = set()
        for fname in docs:
            doc = docs[fname]
            doc_score = program.eval(doc)
            if doc_score == 2:
                top_docs.add(fname)
            elif doc_score == 0:
                bottom_docs.add(fname)
        if config.debug:
            print(len(bottom_docs),len(top_docs))
        if config.use_mutual_information and len(top_docs) > 0:
            size = config.predicates_per_round * config.mutual_info_pool_size
        else:
            size = config.predicates_per_round
        topk = PriorityQueue(size)
        i = 0
        if config.debug:
            print("Examples:",examples)
        for pred in self.predGen(docs,features_set,config.depth):
            i += 1
            if i % 100 == 0 and config.debug:
                #print(i)
                pass
            if pred in program or pred in rejected_preds or self.Not(pred) in program or self.Not(pred) in rejected_preds:
                continue
            
            if len(program) == 0 or not config.use_bins or (len(top_docs) == 0 or len(bottom_docs) == 0):
                user_true = 0
                nonuser_true = 0
                user_false = 0
                nonuser_false = 0
                for fname in docs:
                    doc = docs[fname]
                    if pred.eval(doc):
                        if fname in examples:
                            user_true += 1
                        else:
                            nonuser_true += 1
                    else:
                        if fname in examples:
                            user_false += 1
                        else:
                            nonuser_false += 1
                if user_true + user_false == 0:
                    user_approved = 0
                else:
                    user_approved = user_true / (user_true + user_false)
                full_docs = (user_true + nonuser_true) / (user_true + user_false + nonuser_true + nonuser_false)
                score = user_approved - full_docs
            else:
                top_true = 0
                top_false = 0
                mid_true = 0
                mid_false = 0
                bottom_true = 0
                bottom_false = 0
                for fname in docs:
                    doc = docs[fname]
                    if pred.eval(doc):
                        if fname in top_docs:
                            top_true += 1
                        elif fname in bottom_docs:
                            bottom_true += 1
                        else:
                            mid_true += 1
                    else:
                        if fname in top_docs:
                            top_false += 1
                        elif fname in bottom_docs:
                            bottom_false += 1
                        else:
                            mid_false += 1
                top_score = top_true / (top_true + top_false)
                bottom_score = bottom_true / (bottom_true + bottom_false)
                score = top_score - bottom_score
            item_i = (abs(score),i,pred,score)
            if not topk.full():
                topk.put(item_i)
            else:
                item_k = topk.get()
                to_push = max(item_i,item_k,key=lambda item:item[0])
                topk.put(to_push)
        
        if config.use_mutual_information and len(top_docs) > 0:
            return self.filter_topk(config.predicates_per_round,topk,docs)
        else:
            preds = []
            while not topk.empty():
                item_i = topk.get()
                if config.debug:
                    print(item_i)
                if item_i[3] < 0:
                    preds.append(self.Not(item_i[2]))
                else:
                    preds.append(item_i[2])
            return preds

class ImgLang(Lang):
    class Document:
        def __init__(self,fname,features):
            self.features = set(features)
            self.fname = fname
    class Pred:
        def __repr__(self):
            pass
        def eval(self,doc):
            pass

    class Exists(Pred):
        def __init__(self, feature: str):
            self.feature = feature

        def eval(self,doc):
            return self.feature in doc.features

        def __repr__(self):
            return f"Exists({self.feature})"

        def __eq__(self,other):
            return isinstance(other,ImgLang.Exists) and self.feature == other.feature
        
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
            return isinstance(other,ImgLang.Not) and self.pred == other.pred

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
            return isinstance(other,ImgLang.AndPred) and ((self.pred1 == other.pred1 and self.pred2 == other.pred2) or 
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
            return isinstance(other,ImgLang.OrPred) and ((self.pred1 == other.pred1 and self.pred2 == other.pred2) or 
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

        def __repr__(self):
            return f"Multi([{','.join(self.preds)}],{self.thresh})"

    def parse_pred(self,input_str):
        def parse_exists(tokens):
            return ImgLang.Exists(tokens[1])

        def parse_not(tokens):
            return ImgLang.Not(tokens[1])

        def parse_and(tokens):
            return ImgLang.AndPred(tokens[1], tokens[2])

        def parse_or(tokens):
            return ImgLang.OrPred(tokens[1], tokens[2])

        def parse_multi(tokens):
            return ImgLang.MultiPred(tokens[0][0], int(tokens[0][1]))

        identifier = QuotedString('"', multiline=True, unquoteResults=True) | Word(printables)
        exists_parser = Literal("Exists") + Suppress("(") + identifier + Suppress(")")
        not_parser = Forward()
        and_parser = Forward()
        or_parser = Forward()
        multi_parser = Forward()

        exists_parser.setParseAction(parse_exists)
        not_parser.setParseAction(parse_not)
        and_parser.setParseAction(parse_and)
        or_parser.setParseAction(parse_or)
        multi_parser.setParseAction(parse_multi)

        pred_parser = exists_parser | not_parser | and_parser | or_parser #| multi_parser
        not_parser <<= Literal("Not") + Suppress("(") + pred_parser + Suppress(")")
        and_parser <<= Literal("And") + Suppress("(") + pred_parser + Suppress(",") + pred_parser + Suppress(")")
        or_parser <<= Literal("Or") + Suppress("(") + pred_parser + Suppress(",") + pred_parser + Suppress(")")
        #multi_parser <<= Literal("Multi(") + Group(ZeroOrMore(pred_parser + Suppress(","))) + Literal(",") + Word(nums)

        return pred_parser.parseString(input_str, parseAll=True)[0]
    
    def predGen(self,docs,features,depth):
        def genPredsLevel(depth):
            if depth in self.preds:
                return self.preds[depth]
            elif depth == 0:
                levelPreds = [ImgLang.Exists(feature) for feature in features]
                self.preds[0] = levelPreds
                return levelPreds
            elif depth == 1:
                levelPreds = set()
                for fname in docs:
                    doc = docs[fname]
                    for i, f1 in enumerate(list(doc.features)):
                        for j, f2 in enumerate(list(doc.features)):
                            if i < j:
                                pred = ImgLang.AndPred(ImgLang.Exists(f1),ImgLang.Exists(f2))
                                invPred = ImgLang.AndPred(ImgLang.Exists(f2),ImgLang.Exists(f1))
                                if pred not in levelPreds and invPred not in levelPreds:
                                    levelPreds.add(pred)
                levelPreds = list(levelPreds)
                self.preds[1] = levelPreds
                return self.preds[1]
        return genPredsLevel(depth)

    def displayPred(self,pred):
        print(pred)

class MusicLang(Lang):
    class Document:
        def __init__(self,fname,features):
            self.features = features
            self.fname = fname
    class Pred:
        def __repr__(self):
            pass
        def eval(self,doc):
            pass

    class Not(Pred):
        def __init__(self,pred):
            self.pred = pred

        def eval(self,doc):
            return not self.pred.eval(doc)

        def __repr__(self):
            return f"Not({self.pred})"

        def __eq__(self,other):
            return isinstance(other,MusicLang.Not) and self.pred == other.pred

        def __hash__(self):
            return hash(("Not",self.pred))
        
        def chords(self):
            return self.pred.chords()
        
        def __len__(self):
            return len(self.pred)
        
    class OrPred(Pred):
        def __init__(self,pred1, pred2):
            self.pred1 = pred1
            self.pred2 = pred2
        
        def eval(self,doc):
            return self.pred1.eval(doc) or self.pred2.eval(doc)

        def __repr__(self):
            return f"Or({self.pred1},{self.pred2})"

        def __eq__(self,other):
            return isinstance(other,MusicLang.OrPred) and ((self.pred1 == other.pred1 and self.pred2 == other.pred2) or 
            (self.pred2 == other.pred1 and self.pred1 == other.pred2))

        def __hash__(self):
            return hash(("Or",self.pred1,self.pred2))
        
    class Chord(Pred):
        def __init__(self,name):
            self.name = name

        def eval(self,doc):
            return self.name in doc.features
        
        def chords(self):
            return [self.name]
        
        def __repr__(self):
            return f"Chord({self.name})"
        
        def __eq__(self,other):
            return isinstance(other,MusicLang.Chord) and other.name == self.name
    
        def __hash__(self):
            return hash(("Chord",self.name))
        
        def __len__(self):
            return 1
    
    class ChordSeq(Pred):
        def __init__(self,seq):
            self.seq = seq

        def chords(self):
            return [c.name for c in self.seq]

        def eval(self,doc):
            for i in range(0,len(doc.features)-len(self.seq)):
                if self.chords() == doc.features[i:i+len(self.seq)]:
                    return True
            return False
        
        def __repr__(self):
            return f'Seq({",".join([str(c) for c in self.seq])})'
        
        def __eq__(self,other):
            if not isinstance(other,MusicLang.ChordSeq):
                return False
            if len(self.seq) != len(other.seq):
                return False
            for c1, c2 in zip(self.seq,other.seq):
                if c1 != c2:
                    return False
            return True
        
        def __hash__(self):
            return hash(("ChordSeq",','.join([str(chord) for chord in self.seq])))
        
        def __len__(self):
            return len(self.seq)
        
    def displayPred(self,pred):
        print(pred)
        cwd = os.getcwd()
        create_midi(pred.chords(),"tmp",cwd)
        midi_path = os.path.join(cwd, 'tmp.mid')
        pygame.mixer.music.load(midi_path,namehint="midi")
        pygame.mixer.music.play()
        time.sleep(len(pred))

    def parse_pred(self,input_str):
        if '[' in input_str:
            stripped_str = input_str[1:-1]
            chords = stripped_str.split(',')
            return MusicLang.ChordSeq([MusicLang.Chord(chord) for chord in chords])
        else:
            return MusicLang.Chord(input_str)
    
    def predGen(self,docs,features,depth):
        def genPreds(depth):
            if depth in self.preds:
                return self.preds[depth]
            levelPreds = set()
            if depth == 1:
                for fname in docs:
                    doc = docs[fname]
                    for feature in doc.features:
                        chord = MusicLang.Chord(feature)
                        if chord not in levelPreds:
                            levelPreds.add(chord)
            else:
                for fname in docs:
                    doc = docs[fname]
                    for i in range(len(doc.features)-(depth-1)):
                        seq = MusicLang.ChordSeq([MusicLang.Chord(chord) for chord in doc.features[i:i+depth]])
                        if seq not in levelPreds:
                            levelPreds.add(seq)
            self.preds[depth] = levelPreds
            return levelPreds
        for i in range(1, depth+1):
            for pred in genPreds(i):
                yield pred

            

    
