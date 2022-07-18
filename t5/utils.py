import paddle


def convert_example(example,
                    tokenizer,
                    max_seq_length=128,
                    is_test=False):
    source = example["source"]
    input_ids = tokenizer(source)['input_ids']
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]

    attention_mask = [1] * len(input_ids)

    if not is_test:
        target = example["target"]
        correction_labels = tokenizer(target)['input_ids']
        if len(correction_labels) > max_seq_length:
            correction_labels = correction_labels[:max_seq_length]

        return input_ids, attention_mask, correction_labels
    else:
        return input_ids, attention_mask


def read_train_ds(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            source, target = line.strip('\n').split('\t')[0:2]
            yield {'source': source, 'target': target}


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)