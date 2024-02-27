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

    def __repr__(self):
        return f"Multi([{','.join(self.preds)}],{self.thresh})"

def parse_pred(input_str):
    def parse_exists(tokens):
        return Exists(tokens[1])

    def parse_not(tokens):
        return Not(tokens[1])

    def parse_and(tokens):
        return AndPred(tokens[1], tokens[2])

    def parse_or(tokens):
        return OrPred(tokens[1], tokens[2])

    def parse_multi(tokens):
        return MultiPred(tokens[0][0], int(tokens[0][1]))

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