# News Knowledge Extraction Tool

## 📝 Updates
✅`2023-10-21` NKETool is released.

✅`2024-03-12` Multiple detail improvements and performance optimisations.

## ⚙️ Environment
#### 1. Install python3.9. 
#### 2. Install the requirements:
```
pip install requirements.txt -r
```

</br>

**If you have problems configuring your environment using the above method, follow these steps:**

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

- If the download goes wrong due to the Internet speed:

Visit https://download.pytorch.org/whl/torch_stable.html and download WHL files to install.
 
```
pip install OmniEvent
```

- If deepspeed installation failed:

Visit https://www.piwheels.org/project/deepspeed/ and download deepspeed-0.7.2-py3-none-any.whl to install.

- If sentencepiece versions have conflicting dependences:

Before install OmniEvent, firstly:`pip install sentencepiece==0.1.97`


## 🛠️ How to use NKETool
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

>>> model1_name_or_path = "s2s-mt5-ed"

>>> model2_name_or_path = "s2s-mt5-eae"

>>> device = "cuda"

>>> print(nke_tool(text, model1_name_or_path, model2_name_or_path, device))

[
	{
		'text': '今日沪深两市平开后一路下挫，三大股指均未能形成有效反弹。',
		'events':
			[
				{
					'type': '下跌',
					'offset': [11, 13],
					'trigger': '下挫',
					'arguments':
						[
							{
								'mention': '今日',
								'offset': [0, 2],
								'role': '时间'
							},
							{
								'mention': '沪深两市',
								'offset': [2, 6],
								'role': '跌停股票'
							}
						]
				}
			],
		'sentiment_label': 0.456994,
		'sentiment': 'negative'
	},
	{
		'text': '格力十一年来首次不分红，开盘逼近跌停。',
		'events':
			[
				{
					'type': '跌停',
					'offset': [16, 18],
					'trigger': '跌停',
					'arguments':
						[
							{
								'mention': '格力',
								'offset': [0, 2],
								'role': '股份股权转让-target-company'
							}
						]
				}
			],
		'sentiment_label': 0.001994,
		'sentiment': 'negative'
	}
]
```

- News Knowledge Extraction Tool is a off-the-shelf tool. If you want to train your Own Model with OmniEvent, you can follow: https://github.com/THU-KEG/OmniEvent/tree/main

## 🪴 Acknowledgement
This project is based on the following open source projects, hereby expresses the sincere thanks to the relevant projects and their developers:

- **OmniEvent**: https://github.com/THU-KEG/OmniEvent
- **Cemotion**: https://github.com/Cyberbolt/Cemotion
