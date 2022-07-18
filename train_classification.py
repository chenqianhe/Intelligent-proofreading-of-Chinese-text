from paddlenlp.datasets import load_dataset
import json
import os
import random
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
import functools
import numpy as np
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
import time
import paddle.nn.functional as F
import paddle
from eval import evaluate
import argparse


num_classes = 3


def parse_args():
    parse = argparse.ArgumentParser(description='Train Classification Model')
    parse.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path')

    parse.add_argument('--learning_rate', type=float, required=True, help='Learning Rate')

    parse.add_argument('--train_data_path', type=str, required=True, help='Train data path')
    parse.add_argument('--train_data_is_ground_eval', default=False, type=bool, required=True,
                       help='Train data is ground eval')

    parse.add_argument('--max_seq_length', default=128, type=int, required=True, help='Max Seq Length')
    parse.add_argument('--batch_size', default=64, type=int, required=True, help='Batch Size')

    parse.add_argument('--eval_data_path', type=str, required=True, help='Eval data path')
    parse.add_argument('--eval_data_is_ground_eval', default=False, type=bool, required=True,
                       help='Eval data is ground eval')

    parse.add_argument('--model_save_path', type=str, required=True, help='Model save path')

    parse.add_argument('--epoch', type=int, required=True, help='Epochs')
    parse.add_argument('--print_step', default=10, type=int, required=True, help='Print acc .etc')
    parse.add_argument('--eval_step', default=20, type=int, required=True, help='Eval model')

    args = parse.parse_args()
    return args


# 定义读取数据集函数
def read_custom_data(data_files, is_ground_eval=False):
    pure_data = []
    with open(data_files) as f:
        all_data = json.load(f)
        if not is_ground_eval:
            for data in all_data:
                pure_data.append([data['target'], 0])
                if len(data['source']) != len(data['target']):
                    pure_data.append([data['source'], 2])
                else:
                    pure_data.append([data['source'], 1])
        else:
            for data in all_data:
                if data['type'] == 'positive':
                    pure_data.append([data['target'], 0])
                else:
                    if len(data['source']) != len(data['target']):
                        pure_data.append([data['source'], 2])
                    else:
                        pure_data.append([data['source'], 1]) 

    random.shuffle(pure_data)
    for data in pure_data:
        yield {"text": (data[0]), "labels": data[1]}


# 数据预处理函数，利用分词器将文本转化为整数序列
def preprocess_function(examples, tokenizer, max_seq_length):
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    result["labels"] = examples["labels"]
    return result


if __name__ == "__main__":
    args = parse_args()

    train_ds = load_dataset(read_custom_data, is_test=False, lazy=False, data_files=args.train_data_path, is_ground_eval=args.train_data_is_ground_eval) 
    eval_ds = load_dataset(read_custom_data, is_test=True, lazy=False, data_files=args.eval_data_path, is_ground_eval=args.eval_data_is_ground_eval)

    # fine-tune-eval

    # lazy=False，数据集返回为MapDataset类型
    print("数据类型:", type(train_ds))
    print("训练集样例:", train_ds[0])
    print("验证集样例:", eval_ds[0])

    model_name_or_path = args.model_name_or_path
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_classes=num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func)
    eval_ds = eval_ds.map(trans_func)

    # collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
    collate_fn = DataCollatorWithPadding(tokenizer)

    # 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
    train_batch_sampler = BatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_batch_sampler = BatchSampler(eval_ds, batch_size=args.batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    eval_data_loader = DataLoader(dataset=eval_ds, batch_sampler=eval_batch_sampler, collate_fn=collate_fn)

    optimizer = paddle.optimizer.AdamW(learning_rate=args.learning_rate, parameters=model.parameters())
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    epochs = args.epoch # 训练轮次
    ckpt_dir = args.model_save_path #训练过程中保存模型参数的文件夹

    global_step = 0 #迭代次数
    tic_train = time.time()
    best_score = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.sigmoid(logits)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            # 打印损失函数值、准确率、计算速度
            global_step += 1
            if global_step % args.print_step == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, auc: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                        args.print_step / (time.time() - tic_train)))
                tic_train = time.time()
            
            # 反向梯度回传，更新参数
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            # 评估当前训练的模型、保存当前最佳模型参数和分词器的词表等
            if global_step % args.eval_step == 0:
                save_dir = ckpt_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                eval_score = evaluate(model, criterion, metric, eval_data_loader)
                if eval_score > best_score:
                    best_score = eval_score
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

