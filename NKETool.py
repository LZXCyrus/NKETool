from OmniEvent.infer import infer
import re
from cemotion import Cemotion

c = Cemotion()

def nke_tool(text):
    #split sentences
    sentences = re.split('(。|！|\!|？|\?|\;|；)', text)
    sents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        sents.append(sent)

    results = []

    for j in range(len(sents)):

        # Event Module
        event_results = infer(text=sents[j], task="ED")
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


print(nke_tool("今日沪深两市平开后一路下挫，三大股指均未能形成有效反弹。格力十一年来首次不分红，开盘逼近跌停。"))