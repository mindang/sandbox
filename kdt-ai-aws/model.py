import itertools
from utils import clean_text


class ModelHandler:
    def __init__(self):
        self.id2label = {0: 'negative', 1: 'positive'}

    def _clean_text(self, text):
        model_input = []
        if isinstance(text, str):
            cleaned_text = clean_text(text)
            model_input.append(cleaned_text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and (all(isinstance(t, str) for t in text)):
            cleaned_text = itertools.chain((clean_text(t) for t in text))
            model_input.extend(cleaned_text)
        else:
            model_input.append('')
        return model_input


class MLModelHandler(ModelHandler):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self, ):
        # De-serializing model and loading vectorizer
        import joblib
        self.vectorizer = joblib.load('model/ml_vectorizer.pkl')
        self.model = joblib.load('model/ml_model.pkl')

    def preprocess(self,text):
        # cleansing raw text
        model_input =  self._clean_text(text)
        
        # vectorizing cleaned text
        model_input = self.vectorizer.transform(model_input)
        return model_input
        
    def inference(self, data):
        # get predictions from model as probabilities
        model_output = self.model.predict_proba(data)
        return model_output

    def postprocess(self, model_output):
        # process predictions to predicted label and output format
        predicted_probabilities = model_output.max(axis=1)
        predicted_ids = model_output.argmax(axis=1)
        predicted_labels =[self.id2label[id_] for id_ in predicted_ids]
        return predicted_labels , predicted_probabilities

    def handle(self, text):
        # do above processes
        model_input = self.preprocess(text)
        model_output = self.inference(model_input)
        return  self.postprocess(model_output)

class DLModelHandler(ModelHandler):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self, ):
        # Loading tokenizer and De-serializing model
        ...

    def preprocess(self, ):
        # cleansing raw text
        ...
        
        # vectorizing cleaned text
        ...

    def inference(self, ):
        # get predictions from model as probabilities
        ...
        
    def postprocess(self, ):
        # process predictions to predicted label and output format
        ...

    def handle(self, ):
        ...
