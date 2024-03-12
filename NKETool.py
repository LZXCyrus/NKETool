from inference import infer
import re
import json
import os
from cemotion import Cemotion
from OmniEvent.utils import check_web_and_convert_path
from OmniEvent.model.model import get_model_cls
from transformers import (
    BertTokenizerFast,
    RobertaTokenizerFast,
    T5TokenizerFast,
    MT5TokenizerFast,
    BartTokenizerFast
)

c = Cemotion()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

TOKENIZER_NAME_TO_CLS = {
    "BertTokenizer": BertTokenizerFast,
    "RobertaTokenizer": RobertaTokenizerFast,
    "T5Tokenizer": T5TokenizerFast,
    "MT5Tokenizer": MT5TokenizerFast,
    "BartTokenizer": BartTokenizerFast
}


def get_tokenizer(tokenizer_name_or_path):
    path = check_web_and_convert_path(tokenizer_name_or_path, "tokenizer")
    tokenizer_config = json.load(open(os.path.join(path, "tokenizer_config.json")))
    tokenizer_cls = TOKENIZER_NAME_TO_CLS[tokenizer_config["tokenizer_class"]]
    tokenizer = tokenizer_cls.from_pretrained(path)
    return tokenizer


def get_model(model_args, model_name_or_path):
    path = check_web_and_convert_path(model_name_or_path, "model")
    model = get_model_cls(model_args).from_pretrained(path)
    return model


def get_pretrained(model_name_or_path, device):
    model_args = AttrDict({
        "paradigm": "seq2seq",
        "model_type": "mt5"
    })
    model = get_model(model_args, model_name_or_path)
    model = model.to(device)
    tokenizer = get_tokenizer(model_name_or_path)

    return model, tokenizer


def nke_tool(text,model1_name_or_path,model2_name_or_path,device):
    model1, token1 = get_pretrained(model1_name_or_path, device)
    model2, token2 = get_pretrained(model2_name_or_path, device)
    model = [model1, token1]
    tokenizer = [model2, token2]
    #split sentences
    sentences = re.split('(。|！|\!|？|\?|\;|；)', text)
    sents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        sents.append(sent)

    results = []

    for j in range(len(sents)):

        # Event Module
        event_results = infer(text=sents[j],model=model,tokenizer=tokenizer, task="EE")
        event_results = event_results[0]
        tex = event_results['text']
        event = event_results['events']

        # Sentiment Module
        sentiment_label = c.predict(sents[j])
        if sentiment_label > 0.5:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        result = {'text':tex,'events':event,'sentiment_label':sentiment_label,'sentiment':sentiment}
        results.append(result)

    return results
