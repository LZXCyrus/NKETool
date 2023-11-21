# News Knowledge Extraction Tool


## Environment
#### 1. Install python3.9. 
#### 2. Install the requirements:

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

 <br/>
- If the download goes wrong due to the Internet speed:
 <br/>
Visit https://download.pytorch.org/whl/torch_stable.html and download WHL files to install.
 <br/>
 
```
pip install OmniEvent
```

 <br/>
- If deepspeed installation failed:
 <br/>
Visit https://www.piwheels.org/project/deepspeed/ and download deepspeed-0.7.2-py3-none-any.whl to install.
 <br/>
- If sentencepiece versions have conflicting dependences:
 <br/>
Before install OmniEvent, firstly:

```
pip install sentencepiece==0.1.97
```


## How to use News Knowledge Extraction Tool
#### 1. Download bert-base-chinese
Change the above code to your path in cemotion.py.
```python
>>> tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
```
#### 2. Download pre-trained model **cemotion_2.0.pt**
Visit https://github.com/Cyberbolt/Cemotion/releases/download/2.0/cemotion_2.0.pt and download the model. Then change the above code to your path in cemotion.py.
```python
>>> self.model = load_model(model, 'cemotion_2.0.pt')
```
#### 3. Easy start
*Note that it may take a few minutes to download checkpoint at the first time (approximately 5GB)*.

One example is as follows:
```python
>>> from NKETool import nke_tool

>>> text = "今日沪深两市平开后一路下挫，三大股指均未能形成有效反弹。格力十一年来首次不分红，开盘逼近跌停。"
>>> print(nke_tool(text))
[
	{
		'text': '今日沪深两市平开后一路下挫，三大股指均未能形成有效反弹。', 
		'events': [{'type': '下跌', 'trigger': '下挫', 'offset': [11, 13]}], 'arguments': [
            {   "mention": "今日", "offset": [1, 3], "role": "时间"},
            {   "mention": "沪深两市", "offset": [2, 6], "role": "名称"}],
		'sentiment_label': 0.456995,
		'sentiment': 'negative'
	},
	{
		'text': '格力十一年来首次不分红，开盘逼近跌停。', 
		'events': [{'type': '跌停', 'trigger': '跌停', 'offset': [16, 18]}], 'arguments': [
            {   "mention": "格力", "offset": [1, 3], "role": "公司名称"},
            {   "mention": "十一年", "offset": [2, 5], "role": "时间"}],
		'sentiment_label': 0.001994,
		'sentiment': 'negative'
	}
]
```

- News Knowledge Extraction Tool is a off-the-shelf tool. If you want to train your Own Model with OmniEvent: https://github.com/THU-KEG/OmniEvent/tree/main
