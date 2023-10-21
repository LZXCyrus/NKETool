import os

import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, logging


logging.set_verbosity_error()
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')


class SentimentClassifier(torch.nn.Module):

    def __init__(self, num_classes=1):
        super(SentimentClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese', num_labels=num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs[0]


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


class Cemotion:
    def __init__(self):
        model = SentimentClassifier(num_classes=1)
        self.model = load_model(model, 'cemotion_2.0.pt')
        self.device = torch.device('cpu')

    def predict(self, text):

        if type(text) == type('text mode'):
            list_text = [text]  # 将文本转为列表
            prediction = self.deal(list_text)[0]
            return round(prediction, 6)

        elif type(text) == type(['list mode']) or type(text) == type(np.array(['list mode'])):
            if type(text) == type(np.array(['list mode'])):
                text = text.tolist()

            list_text = text
            prediction = self.deal(list_text)
            list_new = []

            for one, two in zip(list_text, prediction):
                list_new.append([one, round(two, 6)])

            return list_new

    def deal(self, list_text: list) -> list:

        predictions = []
        for sentence in list_text:
            inputs = tokenizer.encode_plus(
                sentence, add_special_tokens=True, max_length=128, padding='max_length', truncation=True)
            input_ids = torch.tensor(
                inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(
                inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device)
            token_type_ids = torch.tensor(
                inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).squeeze(1)
            predict = torch.sigmoid(outputs).cpu().detach().numpy()[0]
            predictions.append(predict)
        return predictions