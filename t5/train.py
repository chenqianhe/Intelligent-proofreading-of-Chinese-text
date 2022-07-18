from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import convert_example, create_dataloader, read_train_ds


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--model_name_or_path", type=str, default="t5-v1_1-large", help="Pretraining model name or path")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--epochs", type=int, default=3, help="Number of epoches for training.")
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Select cpu, gpu devices to train model.")
parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--ignore_label", default=-1, type=int, help="Ignore label for CrossEntropyLoss")
parser.add_argument("--train_ds_dir", default=None, type=str, help="The directory of train dataset.")
parser.add_argument("--eval_ds_dir", default=None, type=str, help="The directory of eval dataset.")

# yapf: enable
args = parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, eval_data_loader):
    model.eval()

    loss = 0

    for step, batch in enumerate(eval_data_loader, start=1):
        input_ids, attention_mask, correction_labels = batch

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=correction_labels)
        loss += output[0]
        logits = output[1]

    loss /= step
    logger.info("Sentence-Level Performance:")
    logger.info("loss: %f" % (loss))

    model.train()
    return loss


def do_train(args):
    set_seed(args)
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    
    # Extend current training dataset by providing extra training
    # datasets directory. The suffix of dataset file name in extra
    # dataset directory has to be ".txt". The data format of
    # dataset need to be a couple of senteces at every line, such as:
    # "城府宫员表示，这是过去三十六小时内第三期强烈的余震。\t政府官员表示，这是过去三十六小时内第三起强烈的余震。\n"
    if args.train_ds_dir is not None and os.path.exists(args.train_ds_dir) and args.eval_ds_dir is not None and os.path.exists(args.eval_ds_dir):
        train_data_files = [
            os.path.join(args.train_ds_dir, data_file)
            for data_file in os.listdir(args.train_ds_dir)
            if data_file.endswith(".txt")
        ]
        train_ds = load_dataset(read_train_ds, 
                                data_path=train_data_files[0], 
                                lazy=False)
        train_data = train_ds.data

        for data_file in train_data_files[1:]:
            t_ds = load_dataset(read_train_ds,
                                data_path=data_file,
                                lazy=False)
            train_data += t_ds.data
        train_ds = MapDataset(train_data)

        eval_data_files = [
            os.path.join(args.eval_ds_dir, data_file)
            for data_file in os.listdir(args.eval_ds_dir)
            if data_file.endswith(".txt")
        ]
        eval_ds = load_dataset(read_train_ds, 
                                data_path=eval_data_files[0], 
                                lazy=False)
        eval_data = eval_ds.data

        for data_file in eval_data_files[1:]:
            e_ds = load_dataset(read_train_ds,
                                data_path=data_file,
                                lazy=False)
            eval_data += e_ds.data
        eval_ds = MapDataset(eval_data)

    else:
        logger.info("Data abnormal")
        return

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=1),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    eval_data_loader = create_dataloader(eval_ds,
                                         mode='eval',
                                         batch_size=args.batch_size,
                                         batchify_fn=batchify_fn,
                                         trans_fn=trans_func)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    logger.info("Total training step: {}".format(num_training_steps))
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    global_steps = 1
    best_loss = 9999999999
    tic_train = time.time()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, attention_mask, correction_labels = batch
            # print(input_ids)
            # print(attention_mask)
            # print(correction_labels)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=correction_labels)
            loss = output[0]
            logits = output[1]

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_steps % args.logging_steps == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_steps, epoch, step, loss, args.logging_steps /
                       (time.time() - tic_train)))
                tic_train = time.time()
            if global_steps % args.save_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    logger.info("Eval:")

                    eval_loss = evaluate(model, eval_data_loader)
                    
                    model_file = "model_%d" % global_steps
                    if eval_loss < best_loss:
                        # save best model
                        os.makedirs(os.path.join(args.output_dir, model_file + "_best"))
                        model.save_pretrained(os.path.join(args.output_dir, model_file + "_best"))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, model_file + "_best"))
                        logger.info(
                            "Save best model at {} step.".format(global_steps))
                        best_loss = eval_loss
                        model_file = model_file + "_best"
            if args.max_steps > 0 and global_steps >= args.max_steps:
                return
            global_steps += 1


if __name__ == "__main__":
    do_train(args)