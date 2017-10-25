# NLU

该项目提供意图识别与命名实体识别两种功能。

## 安装

本项目基于python3，建议在python3环境下运行，python2下可能会有转义问题。

```bash
git clone http://192.168.1.161/nlu/DeepNlu
```

### 安装依赖包

```bash
pip -r requirements.txt
```

命令行工具`main.py`包含以下6个命令提供使用

- evaluate_line：NLU识别
- evaluate_line_ner：NER识别
- evaluate_line_int：意图识别
- prepare_dataset intent | ner：准备训练数据，可以选择NER格式或者意图识别格式
- train_int model_type data_train data_val data_test vocab：意图识别模型训练
- train_ner：NER 模型训练(NER部分参数较为复杂，在`config_file`里进行调节，部分路径需要进行直接指定，具体参考源代码)


## 使用

本项目直接提供NLU功能，我已经将训练好的模型放在了`ckpt`文件夹中，如果需要自己按照自己的数据进行训练的话参考下面各个部分的使用说明。

要使用整个pipeline，调用以下命令。

```bash
python main.py evaluate_line
```

测试如下


```bash
(py3) ➜  NLU_ git:(master) ✗ python main.py evaluate_line
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/rn/qwrx11q558z9ns4llsv85bzw0000gn/T/jieba.cache
Loading model cost 0.907 seconds.
Prefix dict has been built succesfully.
==========================Loading the Intention Classification model....==========================
2017-10-25 15:32:54.120536: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:32:54.126276: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:32:54.126289: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:32:54.126294: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Model loaded..
==========================Loading the NER model....==========================
/Users/xuguodong/anaconda/envs/py3/lib/python3.6/site-packages/tensorflow/python/ops/gradients_impl.py:95: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
2017-10-25 15:33:02,497 - train.log - INFO - Reading model parameters from ckpt/ner.ckpt
请输入要进行识别的句子：我要去北京
TRAVEL
{'string': '我要去北京', 'entities': [{'word': '北京', 'start': 3, 'end': 5, 'type': 'DESTINATION'}]}
请输入要进行识别的句子：街道口到武汉怎么走
NAVIGATION_ROUTE_FROM_X_TO_Y
{'string': '街道口到武汉怎么走', 'entities': [{'word': '街道口', 'start': 0, 'end': 4, 'type': 'FROM_CITY'}, {'word': '武汉', 'start': 4, 'end': 7, 'type': 'TO_CITY'}]}
请输入要进行识别的句子：什么样的人容易得高血压
HEALTH_DISEASE_SUSCEPTIBLE
{'string': '什么样的人容易得高血压', 'entities': [{'word': '高血压', 'start': 8, 'end': 11, 'type': 'DISEASE'}]}
请输入要进行识别的句子：刘德华今年多少岁了
KNOWLEDGE_FOR_WIKIPEDIA_PEOPLE
{'string': '刘德华今年多少岁了', 'entities': [{'word': '刘德华', 'start': 0, 'end': 4, 'type': 'SUBJECT_NAME'}, {'word': '今年', 'start': 3, 'end': 6, 'type': 'YEAR_RANGE'}, {'word': '多少岁', 'start': 5, 'end': 9, 'type': 'ATTRIBUTE'}]}
请输入要进行识别的句子：这本书多少钱
KNOWLEDGE_QUERY
{'string': '这本书多少钱', 'entities': [{'word': '这本书多少钱', 'start': 0, 'end': 6, 'type': 'QUERY'}]}
请输入要进行识别的句子：给我订一张去北京的机票
TRAVEL
{'string': '给我订一张去北京的机票', 'entities': [{'word': '北京', 'start': 6, 'end': 9, 'type': 'DESTINATION'}]}
请输入要进行识别的句子：这首歌是什么
MUSIC_PLAY
{'string': '这首歌是什么', 'entities': [{'word': '这首歌', 'start': 0, 'end': 4, 'type': 'ANAPHOR'}]}
请输入要进行识别的句子：有关支付宝的新闻
NEWS_EVENT
{'string': '有关支付宝的新闻', 'entities': [{'word': '支付宝', 'start': 2, 'end': 6, 'type': 'NEWS_KEYWORDS'}]}
请输入要进行识别的句子：我要怎么做皮皮虾
SEARCH_RECIPES
{'string': '我要怎么做皮皮虾', 'entities': [{'word': '皮皮虾', 'start': 5, 'end': 8, 'type': 'FOOD'}]}
请输入要进行识别的句子：找美女照片
SEARCH_IMAGE
{'string': '找美女照片', 'entities': [{'word': '美女', 'start': 1, 'end': 4, 'type': 'SEARCHQUERY'}, {'word': '照片', 'start': 3, 'end': 5, 'type': 'FILETYPE'}]}
请输入要进行识别的句子：网上找找怎么写简历
SEARCH_DEFAULT
{'string': '网上找找怎么写简历', 'entities': [{'word': '怎么写简历', 'start': 4, 'end': 9, 'type': 'SEARCHQUERY'}]}
请输入要进行识别的句子：
```

## 意图识别

我已经训练好了三个模型，可以在 `ckpt` 文件夹中找到

1. `CNN`
2. `GRU` 版本的 `RNN` 模型
3. `CNN` 与 `GRU` 融合后的模型（实际使用的版本）

以上三个模型的在测试集的效果如下，可以通过`python main.py test_int`查看测试结果

```bash
(py3) ➜  NLU_ git:(master) ✗ python main.py test_int
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/rn/qwrx11q558z9ns4llsv85bzw0000gn/T/jieba.cache
Loading model cost 0.952 seconds.
Prefix dict has been built succesfully.
2017-10-25 15:16:48.240639: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:16:48.240667: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:16:48.240673: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:16:48.240678: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
sequence max length: 86
sequence max length: 70
sequence max length: 74
CNN model accuracy: 96.41071230136681 %
RNN model accuracy: 95.6106234026003 %
Multi model accuracy: 96.73297033003666 %
```

即可以得到以下结果

|模型|测试集上准确率|
|---|---|
|CNN|96.41071230136681 %|
|RNN|95.6106234026003 %|
|多模型融合|96.73297033003666 %|

### 数据预处理


```bash
python main.py prepare_dataset intent data nlu
```

其中 `data` 为原始数据文件夹，`nlu`为输出数据文件名字，该程序会将原始数据处理成85%:5%:10%的训练：验证：测试三个部分。

### 模型训练

目前提供四种模型，

1. CNN
2. RNN（提供LSTM 与 GRU 两种模型，可以在`configuration.py中进行调整`）
3. CNN-inception
4. CNN + RNN 模型融合

将数据利用data_loader函数预处理后，根据自己需求调整训练配置文件`configuration.py`，
使用以下命令进行训练

```bash
 # 训练CNN模型
python train.py train_int cnn data/nlu.train data/nlu.val data/nlu.test data/vocab.txt

# 训练Inception-CNN模型
python train.py train_int rnn raw/nlu.train data/nlu.val data/nlu.test data/vocab.txt 

# 训练RNN模型
python train.py train_int inception_cnn data/nlu.train data/nlu.val data/nlu.test raw/vocab.txt 
```


### 模型测试

```bash
python main.py evaluate_line_int
```

测试结果如下

```bash
(py3) ➜  NLU_ git:(master) ✗ python main.py evaluate_line_int
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/rn/qwrx11q558z9ns4llsv85bzw0000gn/T/jieba.cache
Loading model cost 1.625 seconds.
Prefix dict has been built succesfully.
Loading the model....
2017-10-25 15:10:19.923422: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:10:19.923703: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:10:19.923802: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:10:19.923812: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Model loaded..
请输入要进行意图识别的句子：
给徐国栋打一个电话
CALL_CALLING

请输入要进行意图识别的句子：
我要去街道口
NAVIGATION_ROUTE_FROM_X_TO_Y

请输入要进行意图识别的句子：
从街道口到光谷要怎么走
NAVIGATION_ROUTE_FROM_X_TO_Y

请输入要进行意图识别的句子：
哪一种人容易得高血压疾病
HEALTH_DISEASE_SUSCEPTIBLE

请输入要进行意图识别的句子：
埃菲尔铁塔的高度是多少啊
KNOWLEDGE_FOR_WIKIPEDIA_LOCATION
```


## 命名实体识别

NER采用了字使用 bi-LSTM + CRF 进行，使用IOB进行tagging，如果需要也可以采用IOBSE格式进行标记，目前测试在IOB tagging下的统计结果如下。

```python

2017-10-19 00:08:32,046 - log/train.log - INFO - evaluate:test
2017-10-19 00:09:10,923 - log/train.log - INFO - processed 197849 tokens with 29781 phrases; found: 30168 phrases; correct: 28502.

2017-10-19 00:09:10,923 - log/train.log - INFO - accuracy:  97.09%; precision:  94.48%; recall:  95.71%; FB1:  95.09

2017-10-19 00:09:10,923 - log/train.log - INFO -           ACCOUNT: precision:  99.76%; recall: 100.00%; FB1:  99.88  422

2017-10-19 00:09:10,923 - log/train.log - INFO -             ALBUM: precision:  90.24%; recall:  97.37%; FB1:  93.67  41

2017-10-19 00:09:10,923 - log/train.log - INFO -             ALBUM: precision:  90.24%; recall:  97.37%; FB1:  93.67  41                                                                         [1204/1385]

2017-10-19 00:09:10,923 - log/train.log - INFO -         ALBUMNAME: precision:  93.18%; recall:  96.47%; FB1:  94.80  176

2017-10-19 00:09:10,923 - log/train.log - INFO -        ALBUM_NAME: precision:  98.28%; recall:  98.28%; FB1:  98.28  116

2017-10-19 00:09:10,923 - log/train.log - INFO -           ANAPHOR: precision:  96.30%; recall:  81.25%; FB1:  88.14  27

2017-10-19 00:09:10,923 - log/train.log - INFO -            ARTIST: precision:  95.45%; recall:  68.48%; FB1:  79.75  66

2017-10-19 00:09:10,923 - log/train.log - INFO - ARTIST_OR_COMPOSER: precision:  80.28%; recall:  80.65%; FB1:  80.46  218

2017-10-19 00:09:10,923 - log/train.log - INFO -           ATHLETE: precision:  95.87%; recall:  88.55%; FB1:  92.06  121

2017-10-19 00:09:10,923 - log/train.log - INFO -         ATTRIBUTE: precision:  75.85%; recall:  88.84%; FB1:  81.83  294

2017-10-19 00:09:10,923 - log/train.log - INFO -              CITY: precision:  72.92%; recall:  85.37%; FB1:  78.65  48

2017-10-19 00:09:10,924 - log/train.log - INFO -       COMPANYNAME: precision:  96.74%; recall:  99.55%; FB1:  98.12  919

2017-10-19 00:09:10,924 - log/train.log - INFO -      CONTACTFIELD: precision:  69.20%; recall:  97.74%; FB1:  81.03  250

2017-10-19 00:09:10,924 - log/train.log - INFO -       CONTACTNAME: precision:  98.91%; recall:  98.71%; FB1:  98.81  1934

2017-10-19 00:09:10,924 - log/train.log - INFO -    CONTACTNAMEBCC: precision:  98.51%; recall:  97.06%; FB1:  97.78  67

2017-10-19 00:09:10,924 - log/train.log - INFO -     CONTACTNAMECC: precision:  97.89%; recall:  98.94%; FB1:  98.41  95

2017-10-19 00:09:10,924 - log/train.log - INFO -         DATERANGE: precision:  98.30%; recall:  98.84%; FB1:  98.57  4601

2017-10-19 00:09:10,924 - log/train.log - INFO -         DATE_NAME: precision:  50.00%; recall: 100.00%; FB1:  66.67  16

2017-10-19 00:09:10,924 - log/train.log - INFO -         DATE_TYPE: precision: 100.00%; recall: 100.00%; FB1: 100.00  7

2017-10-19 00:09:10,924 - log/train.log - INFO -        DATE_VALUE: precision:  66.67%; recall: 100.00%; FB1:  80.00  6

2017-10-19 00:09:10,924 - log/train.log - INFO -       DESCRIPTION: precision:  55.13%; recall: 100.00%; FB1:  71.07  78

2017-10-19 00:09:10,924 - log/train.log - INFO -       DESTINATION: precision:  97.49%; recall:  92.77%; FB1:  95.07  1514

2017-10-19 00:09:10,924 - log/train.log - INFO -         DIRECTORY: precision: 100.00%; recall: 100.00%; FB1: 100.00  10

2017-10-19 00:09:10,924 - log/train.log - INFO -           DISEASE: precision:  96.43%; recall:  75.00%; FB1:  84.37  28

2017-10-19 00:09:10,924 - log/train.log - INFO -          DISTRICT: precision:  79.31%; recall:  95.83%; FB1:  86.79  29

2017-10-19 00:09:10,924 - log/train.log - INFO -          DISTRICT: precision:  79.31%; recall:  95.83%; FB1:  86.79  29

2017-10-19 00:09:10,924 - log/train.log - INFO -      EMAILADDRESS: precision: 100.00%; recall: 100.00%; FB1: 100.00  61

2017-10-19 00:09:10,924 - log/train.log - INFO -   EMAILADDRESSBCC: precision: 100.00%; recall: 100.00%; FB1: 100.00  43

2017-10-19 00:09:10,924 - log/train.log - INFO -    EMAILADDRESSCC: precision: 100.00%; recall: 100.00%; FB1: 100.00  17

2017-10-19 00:09:10,924 - log/train.log - INFO -      EMAILSUBJECT: precision:  98.41%; recall:  94.66%; FB1:  96.50  126

2017-10-19 00:09:10,925 - log/train.log - INFO -           END_LOC: precision:  87.14%; recall:  97.60%; FB1:  92.08  140

2017-10-19 00:09:10,925 - log/train.log - INFO -          FILETYPE: precision:  89.55%; recall: 100.00%; FB1:  94.49  67

2017-10-19 00:09:10,925 - log/train.log - INFO -              FOOD: precision:  87.60%; recall:  90.60%; FB1:  89.08  121

2017-10-19 00:09:10,925 - log/train.log - INFO -  FRIENDORRELATIVE: precision:  95.31%; recall:  95.07%; FB1:  95.19  405

2017-10-19 00:09:10,925 - log/train.log - INFO - FRIENDORRELATIVEBCC: precision:  97.92%; recall: 100.00%; FB1:  98.95  48

2017-10-19 00:09:10,925 - log/train.log - INFO - FRIENDORRELATIVECC: precision: 100.00%; recall:  88.89%; FB1:  94.12  16

2017-10-19 00:09:10,925 - log/train.log - INFO -         FROM_CITY: precision:  65.71%; recall:  88.46%; FB1:  75.41  35

2017-10-19 00:09:10,925 - log/train.log - INFO -          FROM_POI: precision:  78.15%; recall:  93.94%; FB1:  85.32  119

2017-10-19 00:09:10,925 - log/train.log - INFO -            GENDER: precision:  50.00%; recall:  50.00%; FB1:  50.00  2

2017-10-19 00:09:10,925 - log/train.log - INFO -             GENRE: precision:  92.42%; recall:  82.43%; FB1:  87.14  132

2017-10-19 00:09:10,925 - log/train.log - INFO -     IDIOM_KEYWORD: precision: 100.00%; recall: 100.00%; FB1: 100.00  5

2017-10-19 00:09:10,925 - log/train.log - INFO -        IDIOM_NAME: precision:  50.00%; recall:  22.22%; FB1:  30.77  4

2017-10-19 00:09:10,925 - log/train.log - INFO -          KEYWORDS: precision:  96.09%; recall:  98.66%; FB1:  97.36  230

2017-10-19 00:09:10,925 - log/train.log - INFO -              LAST: precision:  98.84%; recall: 100.00%; FB1:  99.42  86

2017-10-19 00:09:10,925 - log/train.log - INFO -            LEAGUE: precision:  98.18%; recall: 100.00%; FB1:  99.08  384

2017-10-19 00:09:10,925 - log/train.log - INFO -          LOCATION: precision:  96.91%; recall:  99.09%; FB1:  97.99  2588

2017-10-19 00:09:10,925 - log/train.log - INFO -          LYRICIST: precision: 100.00%; recall:  66.67%; FB1:  80.00  2

2017-10-19 00:09:10,925 - log/train.log - INFO -          LYRICIST: precision: 100.00%; recall:  66.67%; FB1:  80.00  2                                                                          [1118/1385]

2017-10-19 00:09:10,925 - log/train.log - INFO -       MESSAGEBODY: precision:  97.88%; recall:  99.46%; FB1:  98.66  943

2017-10-19 00:09:10,926 - log/train.log - INFO -      MESSAGE_TYPE: precision:  98.49%; recall:  99.31%; FB1:  98.90  1324

2017-10-19 00:09:10,926 - log/train.log - INFO -               NEW: precision:  98.40%; recall:  99.19%; FB1:  98.80  125

2017-10-19 00:09:10,926 - log/train.log - INFO -          NEWSTYPE: precision: 100.00%; recall: 100.00%; FB1: 100.00  8

2017-10-19 00:09:10,926 - log/train.log - INFO -     NEWS_KEYWORDS: precision:  72.73%; recall:  72.73%; FB1:  72.73  22

2017-10-19 00:09:10,926 - log/train.log - INFO -             ORDER: precision:  52.00%; recall:  68.42%; FB1:  59.09  50

2017-10-19 00:09:10,926 - log/train.log - INFO -  ORGANIZATIONNAME: precision:  91.72%; recall:  92.36%; FB1:  92.04  145

2017-10-19 00:09:10,926 - log/train.log - INFO -            ORIGIN: precision:  97.66%; recall:  95.98%; FB1:  96.81  1026

2017-10-19 00:09:10,926 - log/train.log - INFO -       PHONENUMBER: precision:  98.74%; recall:  95.16%; FB1:  96.92  239

2017-10-19 00:09:10,926 - log/train.log - INFO -          PLAYLIST: precision:  66.67%; recall:  60.00%; FB1:  63.16  9

2017-10-19 00:09:10,926 - log/train.log - INFO -    POETRY_DYNASTY: precision:  90.00%; recall:  90.00%; FB1:  90.00  10

2017-10-19 00:09:10,926 - log/train.log - INFO -    POETRY_KEYWORD: precision:  60.00%; recall: 100.00%; FB1:  75.00  5

2017-10-19 00:09:10,926 - log/train.log - INFO -       POETRY_NAME: precision:  90.00%; recall:  75.00%; FB1:  81.82  20

2017-10-19 00:09:10,926 - log/train.log - INFO -       POETRY_POET: precision: 100.00%; recall:  80.00%; FB1:  88.89  4

2017-10-19 00:09:10,926 - log/train.log - INFO -               POI: precision:  20.00%; recall:   4.76%; FB1:   7.69  5

2017-10-19 00:09:10,926 - log/train.log - INFO -          PROVIDER: precision:  83.69%; recall:  98.98%; FB1:  90.70  233

2017-10-19 00:09:10,926 - log/train.log - INFO -          PROVINCE: precision:  58.33%; recall:  93.33%; FB1:  71.79  24

2017-10-19 00:09:10,926 - log/train.log - INFO -             QUERY: precision:  89.90%; recall:  45.41%; FB1:  60.34  99

2017-10-19 00:09:10,926 - log/train.log - INFO -       SEARCHQUERY: precision:  89.85%; recall:  65.43%; FB1:  75.72  335

2017-10-19 00:09:10,927 - log/train.log - INFO -         SITUATION: precision:  83.33%; recall:  80.00%; FB1:  81.63  24

2017-10-19 00:09:10,927 - log/train.log - INFO -              SONG: precision:  92.00%; recall:  38.98%; FB1:  54.76  25

2017-10-19 00:09:10,927 - log/train.log - INFO -            SOURCE: precision: 100.00%; recall:  90.48%; FB1:  95.00  19

2017-10-19 00:09:10,927 - log/train.log - INFO -            SOURCE: precision: 100.00%; recall:  90.48%; FB1:  95.00  19

2017-10-19 00:09:10,927 - log/train.log - INFO -      SOURCE_QUERY: precision:  79.66%; recall:  83.93%; FB1:  81.74  59

2017-10-19 00:09:10,927 - log/train.log - INFO -       SOURCE_UNIT: precision:  82.48%; recall:  77.93%; FB1:  80.14  137

2017-10-19 00:09:10,927 - log/train.log - INFO -      SOURCE_VALUE: precision:  85.84%; recall:  85.84%; FB1:  85.84  113

2017-10-19 00:09:10,927 - log/train.log - INFO -             SPORT: precision:  93.21%; recall:  94.38%; FB1:  93.79  162

2017-10-19 00:09:10,927 - log/train.log - INFO -        SPORTSTEAM: precision:  96.60%; recall:  98.63%; FB1:  97.61  971

2017-10-19 00:09:10,927 - log/train.log - INFO -         START_LOC: precision:  94.31%; recall:  98.31%; FB1:  96.27  123

2017-10-19 00:09:10,927 - log/train.log - INFO -             STATE: precision:  64.55%; recall: 100.00%; FB1:  78.45  110

2017-10-19 00:09:10,927 - log/train.log - INFO -           STATION: precision:  66.67%; recall:  77.78%; FB1:  71.79  21

2017-10-19 00:09:10,927 - log/train.log - INFO -         STATISTIC: precision:  95.75%; recall:  98.50%; FB1:  97.10  823

2017-10-19 00:09:10,927 - log/train.log - INFO -        STOCKINDEX: precision:  99.85%; recall:  99.85%; FB1:  99.85  1339

2017-10-19 00:09:10,927 - log/train.log - INFO -         STOCKTERM: precision:  95.80%; recall:  99.65%; FB1:  97.69  1785

2017-10-19 00:09:10,927 - log/train.log - INFO -      SUBJECT_NAME: precision:  64.21%; recall:  97.46%; FB1:  77.42  299

2017-10-19 00:09:10,927 - log/train.log - INFO -           SURNAME: precision: 100.00%; recall:  66.67%; FB1:  80.00  4

2017-10-19 00:09:10,927 - log/train.log - INFO -            TARGET: precision:  79.75%; recall:  96.92%; FB1:  87.50  79

2017-10-19 00:09:10,927 - log/train.log - INFO -       TARGET_UNIT: precision:  80.62%; recall:  71.72%; FB1:  75.91  129

2017-10-19 00:09:10,927 - log/train.log - INFO -   TICKET_CATEGORY: precision: 100.00%; recall: 100.00%; FB1: 100.00  44

2017-10-19 00:09:10,928 - log/train.log - INFO -       TICKET_SEAT: precision: 100.00%; recall:  97.73%; FB1:  98.85  86

2017-10-19 00:09:10,928 - log/train.log - INFO -       TICKET_TYPE: precision:  85.87%; recall: 100.00%; FB1:  92.40  92

2017-10-19 00:09:10,928 - log/train.log - INFO -         TIMERANGE: precision:  93.23%; recall:  97.99%; FB1:  95.55  1727

2017-10-19 00:09:10,928 - log/train.log - INFO -             TITLE: precision:  69.15%; recall:  72.40%; FB1:  70.74  402

2017-10-19 00:09:10,928 - log/train.log - INFO -        TOURNAMENT: precision:  99.66%; recall: 100.00%; FB1:  99.83  297

2017-10-19 00:09:10,928 - log/train.log - INFO -        TOURNAMENT: precision:  99.66%; recall: 100.00%; FB1:  99.83  297                                                                        [1032/1385]

2017-10-19 00:09:10,928 - log/train.log - INFO -           TO_CITY: precision:  88.37%; recall:  92.12%; FB1:  90.21  172

2017-10-19 00:09:10,928 - log/train.log - INFO -            TO_POI: precision:  80.89%; recall:  82.25%; FB1:  81.56  361

2017-10-19 00:09:10,928 - log/train.log - INFO -       TO_PROVINCE: precision:  63.64%; recall:  87.50%; FB1:  73.68  11

2017-10-19 00:09:10,928 - log/train.log - INFO -       TRANSITTYPE: precision:  92.78%; recall:  98.90%; FB1:  95.74  194

2017-10-19 00:09:10,928 - log/train.log - INFO -            UNREAD: precision:  96.57%; recall: 100.00%; FB1:  98.25  204

2017-10-19 00:09:10,928 - log/train.log - INFO -        YEAR_RANGE: precision:  50.00%; recall: 100.00%; FB1:  66.67  4

2017-10-19 00:09:10,928 - log/train.log - INFO -              year: precision:  91.67%; recall: 100.00%; FB1:  95.65  12

```

### 数据预处理

```bash
python main.py prepare_dataset ner data nlu
```

其中 `data` 为原始数据文件夹，`nlu`为输出数据文件名字，该程序会将原始数据处理成85%:5%:10%的训练：验证：测试三个部分。

### 模型训练

```bash
 # 训练
python main.py train_ner
```

整个训练过程比较长，建议在GPU环境下训练。

### 模型测试

我已经训练好模型在`ckpt`文件夹下，使用`python main.py evaluate_line_ner`进行测试

```bash
(py3) ➜  NLU_ git:(master) ✗ python main.py evaluate_line_ner
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/rn/qwrx11q558z9ns4llsv85bzw0000gn/T/jieba.cache
Loading model cost 0.949 seconds.
Prefix dict has been built succesfully.
2017-10-25 15:46:36.521827: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:46:36.521862: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:46:36.521874: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 15:46:36.521881: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
/Users/xuguodong/anaconda/envs/py3/lib/python3.6/site-packages/tensorflow/python/ops/gradients_impl.py:95: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
2017-10-25 15:46:40,288 - train.log - INFO - Reading model parameters from ckpt/ner.ckpt
请输入测试句子:给徐国栋打一个电话
{'string': '给徐国栋打一个电话', 'entities': [{'word': '徐国栋', 'start': 1, 'end': 5, 'type': 'CONTACTNAME'}]}
请输入测试句子:介绍一下薛国栋
{'string': '介绍一下薛国栋', 'entities': [{'word': '薛国栋', 'start': 4, 'end': 7, 'type': 'ATHLETE'}]}
请输入测试句子:男子个人争先赛的前三名是谁
{'string': '男子个人争先赛的前三名是谁', 'entities': [{'word': '男子个人争先赛', 'start': 0, 'end': 8, 'type': 'SPORT'}]}
请输入测试句子: 查询上午桂林到武汉的往返的机票
{'string': ' 查询上午桂林到武汉的往返的机票', 'entities': [{'word': '上午', 'start': 3, 'end': 6, 'type': 'TIMERANGE'}, {'word': '桂林', 'start': 5, 'end': 8, 'type': 'START_LOC'}, {'word': '武汉', 'start': 8, 'end': 11, 'type': 'END_LOC'}, {'word': '往返', 'start': 11, 'end': 14, 'type': 'TICKET_TYPE'}]}
请输入测试句子:帮我查下下周一上午从北京到深圳单程的机票
{'string': '帮我查下下周一上午从北京到深圳单程的机票', 'entities': [{'word': '下周一', 'start': 4, 'end': 8, 'type': 'DATERANGE'}, {'word': '上午', 'start': 7, 'end': 10, 'type': 'TIMERANGE'}, {'word': '北京', 'start': 10, 'end': 13, 'type': 'START_LOC'}, {'word': '深圳', 'start': 13, 'end': 16, 'type': 'END_LOC'}, {'word': '单程', 'start': 15, 'end': 18, 'type': 'TICKET_TYPE'}]}
请输入测试句子:下周一晚上北京到广州单程的航班
{'string': '下周一晚上北京到广州单程的航班', 'entities': [{'word': '下周一', 'start': 0, 'end': 4, 'type': 'DATERANGE'}, {'word': '晚上', 'start': 3, 'end': 6, 'type': 'TIMERANGE'}, {'word': '北京', 'start': 5, 'end': 8, 'type': 'START_LOC'}, {'word': '广州', 'start': 8, 'end': 11, 'type': 'END_LOC'}, {'word': '单程', 'start': 10, 'end': 13, 'type': 'TICKET_TYPE'}]}
请输入测试句子:武汉下周一到昆明的飞机票
{'string': '武汉下周一到昆明的飞机票', 'entities': [{'word': '武汉', 'start': 0, 'end': 3, 'type': 'START_LOC'}, {'word': '下周一', 'start': 2, 'end': 6, 'type': 'DATERANGE'}, {'word': '昆明', 'start': 6, 'end': 9, 'type': 'END_LOC'}]}
请输入测试句子:
```


