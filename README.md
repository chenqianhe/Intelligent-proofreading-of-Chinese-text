# 文本智能校对大赛初赛Baseline

>>[赛事链接：文本智能校对大赛](https://aistudio.baidu.com/aistudio/competition/detail/404/0/introduction)

# 一、数据集介绍

[数据集下载](https://aistudio.baidu.com/aistudio/datasetdetail/157348)


```python
# 遍历数据文件夹
!ls -l /home/aistudio/all_data/preliminary_a_data/
```

    total 368228
    -rw-r--r-- 1 aistudio aistudio    214928 Jul 15 15:14 preliminary_a_test_source.json
    -rw-r--r-- 1 aistudio aistudio    400324 Jul 15 15:14 preliminary_extend_train.json
    -rw-r--r-- 1 aistudio aistudio 376015154 Jul 15 15:14 preliminary_train.json
    -rw-r--r-- 1 aistudio aistudio    423209 Jul 15 15:14 preliminary_val.json
    -rw-r--r-- 1 aistudio aistudio      1381 Jul 15 15:14 README.md


- preliminary_train：伪数据约100w, 均为负样本
- preliminary_extend_train: 真实场景训练数据约1000条, 均为负样本
- preliminary_val：真实场景下验证集约1000条(包括约500条正样本和500条负样本）
- preliminary_a_test_source: 真实场景下测试集约1000条（包括约500条正样本和500条负样本）


# 二、思路

我们可以把数据分为正确数据（即不需要纠错），错别字，语义错误，错别字+语义错误4类。

简单来看可以把纠错后与纠错前等长的视为错别字，不等长的视为语义错误；而两者混合错误不好判断，暂不考虑。

但是按上面做法可以发现：
1. 字数相同的也会出现语义错误，这种是颠倒语序造成的语义错误
2. 语义错误很多是字词重复

# 三、做法

1. 分类数据
2. 对识别为错字的先进行错字纠错；无变化再进行语义错误纠错
3. 语义错误纠错包含两部分：

	3.1 算法识别重复内容去除

	3.2 语义纠错模型纠错
   

**本文作者训练的模型已上传到数据集中，[文本智能校对预训练模型](https://aistudio.baidu.com/aistudio/datasetdetail/158748)**

# 三、训练分类模型

* 在nezha模型、ernie模型或其他模型的基础上，首先使用preliminary_train作为训练集，preliminary_extend_train作为验证集训练模型，该阶段模型保存在*_ckpt文件夹中

* 再在训练后模型基础上，使用preliminary_extend_train作为训练集，preliminary_val作为验证集训练模型，该阶段模型保存在*_ft_ckpt文件夹中

* 最后在上一步模型基础上，使用preliminary_val作为训练集训练模型，该阶段模型保存在*_el_ckpt文件夹中


```python
!pip install --upgrade paddlenlp
```


```python
!python train_classification.py --model_name_or_path nezha_el_ckpt --learning_rate 2e-5\
 --train_data_path all_data/preliminary_a_data/preliminary_val.json --train_data_is_ground_eval True\
 --eval_data_path all_data/preliminary_a_data/preliminary_val.json --eval_data_is_ground_eval True\
 --max_seq_length 128 --batch_size 32\
 --model_save_path nezha_el_ckpt --epoch 30 --print_step 10 --eval_step 20
```

## 预测测试集的type


```python
import json
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from eval import predict
```


```python
model_name = "nezha_el_ckpt"
num_classes = 3
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=num_classes)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```



```python
resule_save = []
with open('all_data/preliminary_a_data/preliminary_a_test_source.json') as f:
    test_raw_data = json.load(f)
test_data = [{'text': data['source']} for data in test_raw_data]
label_map = {0: 'Positive', 1: 'Misspelled words', 2: 'Semantic Error'}

results = predict(model, test_data, tokenizer, label_map, batch_size=32)
for idx, text in enumerate(test_data):
    resule_save.append({'source': text['text'], 'type': results[idx], 'id': test_raw_data[idx]['id']})
    print('Data: {} \t Lable: {}'.format(text, results[idx]))
```


```python
print(len(resule_save))
with open('preliminary_a_test_source_with_type.json', 'w') as f:
    json.dump(resule_save, f)
```

    1019


# 四、分离两种错误数据


```python
import json
```


```python
misspell_words_data = ''
semantic_error_data = ''

with open('all_data/preliminary_a_data/preliminary_train.json') as f:
    all_data = json.load(f)
    for data in all_data:
        if len(data['source']) == len(data['target']):
            misspell_words_data += data['source'] + '\t' + data['target'] + '\n'
        else:
            semantic_error_data += data['source'] + '\t' + data['target'] + '\n'

```


```python
with open('all_data/preliminary_a_misspelled_words_data/preliminary_train.txt', 'w') as f:
    f.write(misspell_words_data)
```


```python
with open('all_data/preliminary_a_semantic_error_data/preliminary_train.txt', 'w') as f:
    f.write(semantic_error_data)
```


```python
misspell_words_data = ''
semantic_error_data = ''

with open('all_data/preliminary_a_data/preliminary_extend_train.json') as f:
    all_data = json.load(f)
    for data in all_data:
        if len(data['source']) == len(data['target']):
            misspell_words_data += data['source'] + '\t' + data['target'] + '\n'
        else:
            semantic_error_data += data['source'] + '\t' + data['target'] + '\n'
```


```python
with open('all_data/preliminary_a_misspelled_words_data/preliminary_extend_train.txt', 'w') as f:
    f.write(misspell_words_data)
```


```python
with open('all_data/preliminary_a_semantic_error_data/preliminary_extend_train.txt', 'w') as f:
    f.write(semantic_error_data)
```


```python
misspell_words_data = ''
semantic_error_data = ''

with open('all_data/preliminary_a_data/preliminary_val.json') as f:
    all_data = json.load(f)
    for data in all_data:
        if data['type'] == 'negative':
            if len(data['source']) == len(data['target']):
                misspell_words_data += data['source'] + '\t' + data['target'] + '\n'
            else:
                semantic_error_data += data['source'] + '\t' + data['target'] + '\n'
```


```python
with open('all_data/preliminary_a_misspelled_words_data/preliminary_val.txt', 'w') as f:
    f.write(misspell_words_data)
```


```python
with open('all_data/preliminary_a_semantic_error_data/preliminary_val.txt', 'w') as f:
    f.write(semantic_error_data)
```

# 五、ERNIE-CSC模型 纠正错字

## 训练


```python
%cd ernie_csc
```

    /home/aistudio/ernie_csc



```python
!pip install -r requirements.txt
```


```python
! python download.py --data_dir ./extra_train_ds/ --url https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml
```

    100%|█████████████████████████████████████| 22934/22934 [17:17<00:00, 22.10it/s]



```python
! python change_sgml_to_txt.py -i extra_train_ds/train.sgml -o extra_train_ds/train.txt
```


```python
!python train.py --batch_size 40 --logging_steps 100 --epochs 5 --learning_rate 5e-5 --max_seq_length 128\
 --model_name_or_path ernie-3.0-xbase-zh\
 --full_model_path ../ernie_csc_pre_ckpt/best_model.pdparams\
 --output_dir ../ernie_csc_ft_ckpt/ --extra_train_ds_dir ./extra_train_ds
```

## 导出


```python
!python export_model.py --model_name_or_path ernie-3.0-xbase-zh --params_path ../ernie_csc_pre_ckpt/best_model.pdparams --output_path ../ernie_csc_infer_model/static_graph_params
```



## 预测


```python
from paddlenlp.transformers import ErnieTokenizer
from predict import Predictor
from paddlenlp.data import Vocab

tokenizer = ErnieTokenizer.from_pretrained('ernie-3.0-xbase-zh')
pinyin_vocab = Vocab.load_vocabulary('./pinyin_vocab.txt',
                                     unk_token='[UNK]',
                                     pad_token='[PAD]')
predictor = Predictor('../ernie_csc_infer_model/static_graph_params.pdmodel',
                      '../ernie_csc_infer_model/static_graph_params.pdiparams', 
                      'gpu', 128, tokenizer, pinyin_vocab)

samples = [
    '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
    '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
]

results = predictor.predict(samples, batch_size=2)
for source, target in zip(samples, results):
    print("Source:", source)
    print("Target:", target)
```




    Source: 遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。
    Target: 遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。
    Source: 人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。
    Target: 人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。




# 六、T5模型 纠正语法

## 训练


```python
%cd t5
```

    /home/aistudio/t5



```python
!python train.py --batch_size 32 --logging_steps 100 --epochs 5 --learning_rate 5e-5 --max_seq_length 128\
 --model_name_or_path Langboat/mengzi-t5-base\
 --output_dir ../t5_ft_ckpt/ --train_ds_dir ../all_data/preliminary_a_semantic_error_data --eval_ds_dir ../all_data/preliminary_a_semantic_error_data/eval
```

## 预测


```python
import paddle
from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer
import json
```


```python
tokenizer = T5Tokenizer.from_pretrained('Langboat/mengzi-t5-base')
model = T5ForConditionalGeneration.from_pretrained('../t5_ft_ckpt/model_38000_best')
```



```python
res_t5_pre = []
with open('../all_data/preliminary_a_data/preliminary_a_test_source.json') as f:
    all_data = json.load(f)
    for data in all_data:
        text = data['source']
        inputs = tokenizer(text)
        inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
        output = model.generate(**inputs)
        gen_text = tokenizer.decode(list(output[0].numpy()[0]), skip_special_tokens=True)
        gen_text = gen_text.replace(',', '，')
        gen_text.replace(gen_text[-3:], text[text.rfind(gen_text[-3:]):])
        gen_text = gen_text if text.rfind(gen_text[-3:]) == -1 else gen_text + text[text.rfind(gen_text[-3:])+3:]
        res_t5_pre.append({'inference': gen_text, 'id': data['id']})
```


```python
print(len(res_t5_pre))
```

    1019



```python
with open('../preliminary_a_test_inference.json', 'w') as f:
    json.dump(res_t5_pre, f, ensure_ascii=False)
```

# 七、去重

不考虑效率就用了比较慢的循环方法


```python
def remove_duplication(text: str):
    length = len(text)
    for i in range(length):
        cp_range = min(i+1, length-i-1)
        j = 0
        flag = False
        for j in range(cp_range):
            if text[i-j:i+1] == text[i+1:i+1+j+1]:
                flag = True
                break
        if flag:
            return text.replace(text[i-j:i+1], '', 1)

    return text
```


```python
remove_duplication('我爱爱你')
```




    '我爱你'



# 八、综合

## 首先使路径在 '/home/aistudio'，然后运行分类模型，对测试数据进行分类


```python
import json
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from eval import predict

model_name = "nezha_el_ckpt"
num_classes = 3
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=num_classes)
tokenizer = AutoTokenizer.from_pretrained(model_name)

resule_save = []
with open('all_data/preliminary_a_data/preliminary_a_test_source.json') as f:
    test_raw_data = json.load(f)
test_data = [{'text': data['source']} for data in test_raw_data]
label_map = {0: 'Positive', 1: 'Misspelled words', 2: 'Semantic Error'}

results = predict(model, test_data, tokenizer, label_map, batch_size=32)
for idx, text in enumerate(test_data):
    resule_save.append({'source': text['text'], 'type': results[idx], 'id': test_raw_data[idx]['id']})
    print('Data: {} \t Lable: {}'.format(text, results[idx]))

print(len(resule_save))
with open('all_data/preliminary_a_test_source_with_type.json', 'w') as f:
    json.dump(resule_save, f)
```

    [2022-07-19 00:50:21,887] [    INFO] - We are using <class 'paddlenlp.transformers.nezha.modeling.NeZhaForSequenceClassification'> to load 'nezha_el_ckpt'.
    [2022-07-19 00:50:26,258] [    INFO] - We are using <class 'paddlenlp.transformers.nezha.tokenizer.NeZhaTokenizer'> to load 'nezha_el_ckpt'.


    1019

## 然后定义ERNIE-CSC错别字纠正函数


```python
%cd ernie_csc/
from paddlenlp.transformers import ErnieTokenizer
from predict import Predictor
from paddlenlp.data import Vocab
%cd ../

ernie_tokenizer = ErnieTokenizer.from_pretrained('ernie-3.0-xbase-zh')
pinyin_vocab = Vocab.load_vocabulary('ernie_csc/pinyin_vocab.txt',
                                     unk_token='[UNK]',
                                     pad_token='[PAD]')
ernie_predictor = Predictor('ernie_csc_infer_model/static_graph_params.pdmodel',
                            'ernie_csc_infer_model/static_graph_params.pdiparams', 
                            'gpu', 128, ernie_tokenizer, pinyin_vocab)


def ernie_predict(samples: list, batch_size=2) -> list:
    results = ernie_predictor.predict(samples, batch_size=batch_size)
    
    return results


# 使用示例
# samples = [
#     '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
#     '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
# ]
# results = ernie_predict(samples)
# for source, target in zip(samples, results):
#     print("Source:", source)
#     print("Target:", target)
```




    Source: 遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。
    Target: 遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。
    Source: 人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。
    Target: 人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。




## 再定义T5纠正函数


```python
import paddle
from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer

T5_tokenizer = T5Tokenizer.from_pretrained('Langboat/mengzi-t5-base')
T5_model = T5ForConditionalGeneration.from_pretrained('./t5_ft_ckpt/model_38000_best')

def T5_predict(samples: list) -> list:
    res_t5_pre = []
    for text in samples:
        inputs = T5_tokenizer(text)
        inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
        output = T5_model.generate(**inputs)
        gen_text = T5_tokenizer.decode(list(output[0].numpy()[0]), skip_special_tokens=True)
        gen_text = gen_text.replace(',', '，')
        # 这里是补充生成文本不完整，可能会确实后半部分文本
        gen_text.replace(gen_text[-3:], text[text.rfind(gen_text[-3:]):])
        gen_text = gen_text if text.rfind(gen_text[-3:]) == -1 else gen_text + text[text.rfind(gen_text[-3:])+3:]
        res_t5_pre.append(gen_text)
    
    return res_t5_pre


# 使用示例
samples = [
    '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
    '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
]
results = T5_predict(samples)
for source, target in zip(samples, results):
    print("Source:", source)
    print("Target:", target)
```


    Source: 遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。
    Target: 遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。
    Source: 人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。
    Target: 人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。


## 定义去重函数


```python
def remove_duplication(text: str):
    length = len(text)
    for i in range(length):
        cp_range = min(i+1, length-i-1)
        j = 0
        flag = False
        for j in range(cp_range):
            if text[i-j:i+1] == text[i+1:i+1+j+1]:
                flag = True
                break
        if flag:
            return text.replace(text[i-j:i+1], '', 1)

    return text
```

## 预测结果


```python
import json
res = []

with open('all_data/preliminary_a_test_source_with_type.json') as f:
    raw_data = json.load(f)

    # 这里为了简单，就没有把同一种模型的调用合并成一个list
    for data in raw_data:
        if data['type'] == 'Positive':
            res.append({'inference': data['source'], 'id': data['id']})
        elif data['type'] == 'Misspelled words':
            csc_res = ernie_predict([data['source']])[0]
            if csc_res == data['source']:
                remove_res = remove_duplication(csc_res)
                if remove_res == csc_res:
                    t5_res = T5_predict([csc_res])[0]
                    res.append({'inference': t5_res, 'id': data['id']})
                else:
                    res.append({'inference': remove_res, 'id': data['id']})
            else:
                res.append({'inference': csc_res, 'id': data['id']})
        else:
            remove_res = remove_duplication(data['source'])
            if remove_res == csc_res:
                t5_res = T5_predict([csc_res])[0]
                res.append({'inference': t5_res, 'id': data['id']})
            else:
                res.append({'inference': remove_res, 'id': data['id']})

```


```python
print(len(res))
with open('preliminary_a_test_inference.json', 'w') as f:
    json.dump(res, f)
```

    1019

