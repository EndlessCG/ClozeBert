import copy
import json
import os
import re

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from ClozeDataset import ClozeDataset

bert_path = 'albert-large-v2'
data_path = 'ELE'
loader_path = 'loaders'
WORD_SPLIT_MASKS = '.?!, '
EOS_MARKS = r'[\.?!]'
STRIP_MARKS = '\"\''
DO_STRIP = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
BATCH_SIZE = 40
SENTENCE_LEN = 512
USE_BERT_LM_LOSS = True
MODEL_FILE_NAME = 'cloze_albert_first_attempt.bin'
print(bert_path)
tokenizer = AutoTokenizer.from_pretrained(bert_path)
BLANK_ID = tokenizer.convert_tokens_to_ids('[MASK]')


def plot_train_len(train_data):
    lens = list(map(lambda x: len(x), train_data))
    sns.displot(x=lens)
    plt.show()


def read_data_json_for_whole_passage(dir_name):
    print('Reading data in {}'.format(dir_name))
    data_dir = os.path.join(os.getcwd(), data_path, dir_name)
    file_ls = os.listdir(data_dir)
    options, answers = [], []
    all_masked_indices, all_complete_ids = [], []
    input_ids, input_types, input_masks = [], [], []
    all_raw_options, all_raw_answers = [], []
    for file_name in tqdm(file_ls):
        with open(os.path.join(data_dir, file_name), encoding='utf-8') as f:
            json_content = json.load(f)
            json_content['article'] = json_content['article'].replace('_', '[MASK]')
            encode_dict = tokenizer.encode_plus(json_content['article'])
            # @TODO: try filling all samples to SENTENCE_LEN
            if len(encode_dict['input_ids']) > SENTENCE_LEN:
                cut_split_article = [
                    torch.LongTensor(encode_dict['input_ids'][x * SENTENCE_LEN:(x + 1) * SENTENCE_LEN]).unsqueeze(0) for
                    x in range(len(encode_dict['input_ids']) // SENTENCE_LEN)]
                cut_split_article.append(torch.LongTensor(encode_dict['input_ids'][-SENTENCE_LEN - 1:-1]).unsqueeze(0))
                input_ids.extend(cut_split_article)
                acc = 0
                for segment in cut_split_article:
                    count = segment.shape[1] - (segment - BLANK_ID).count_nonzero()
                    last_index = min(acc + count, len(json_content['options']))
                    all_raw_options.append(json_content['options'][last_index - count:last_index])
                    all_raw_answers.append(json_content['answers'][last_index - count:last_index])
                    acc = acc + count

                    input_mask = torch.zeros(SENTENCE_LEN, dtype=torch.int64)
                    input_mask[0:len(segment)] = 1
                    input_type = torch.ones(SENTENCE_LEN, dtype=torch.int64)
                    input_types.append(input_type.unsqueeze(0))
                    input_masks.append(input_mask.unsqueeze(0))

            else:
                # encode_dict =
                input_ids.append(torch.LongTensor(encode_dict['input_ids']).unsqueeze(0))
                input_types.append(torch.LongTensor(encode_dict['token_type_ids']).unsqueeze(0))
                input_masks.append(torch.LongTensor(encode_dict['attention_mask']).unsqueeze(0))
                all_raw_options.append(json_content['options'])
                all_raw_answers.append(json_content['answers'])

    print('Processing data in {}'.format(dir_name))
    for input_id, r_options, r_answers in zip(tqdm(input_ids), all_raw_options, all_raw_answers):
        option = r_options
        answer = list(map(lambda x: ord(x) - ord('A'), r_answers))

        options_now = []
        for o in option:
            ids = [tokenizer.encode(x)[1] for x in o]
            options_now.append(ids)
        options.append(torch.LongTensor(options_now).to(DEVICE))

        input_id = torch.LongTensor(input_id)
        orig_input_id = copy.deepcopy(input_id)
        if dir_name != 'test':
            ctr = 0
            for i in range(len(input_id[0])):
                if input_id[0][i] == BLANK_ID:
                    input_id[0][i] = tokenizer.encode(option[ctr][answer[ctr]])[1]
                    ctr = ctr + 1

            complete_ids = input_id.masked_fill(
                orig_input_id != BLANK_ID, -100)
            answers.append(torch.unsqueeze(torch.LongTensor(answer), 1).to(DEVICE))

            all_complete_ids.append(complete_ids.to(DEVICE))
        # all_masked_indices.append(
        #     torch.nonzero(torch.LongTensor(complete_ids) != -100)[:, 1].to(DEVICE))
    return input_ids, input_types, input_masks, options, answers, all_complete_ids


def read_data_json(dir_name):
    print('Reading data in {}'.format(dir_name))
    data_dir = os.path.join(os.getcwd(), data_path, dir_name)
    file_ls = os.listdir(data_dir)
    options, answers = [], []
    all_masked_indices, all_complete_ids = [], []
    input_ids, input_types, input_masks = [], [], []
    masked_sentences = []

    for file_name in tqdm(file_ls):
        with open(os.path.join(data_dir, file_name), encoding='utf-8') as f:
            input_id, input_type, input_mask = [], [], []
            mask_indices, sentence_indices = [], []
            complete_ids = []
            json_content = json.load(f)
            orig_article = json_content['article']  # .replace('_', '[MASK]')
            if DO_STRIP:
                orig_article.strip(STRIP_MARKS)
            sentences = re.split(EOS_MARKS, orig_article)
            try:
                sentences.remove('')
            except ValueError:
                pass
            option = json_content['options']
            options_now = []
            for o in option:
                ids = [tokenizer.encode(x)[1] for x in o]
                options_now.append(ids)
            options.append(torch.LongTensor(options_now).to(DEVICE))
            answer = list(map(lambda x: ord(x) - ord('A'), json_content['answers']))
            answers.append(torch.unsqueeze(torch.LongTensor(answer), 1).to(DEVICE))
            ctr = 0
            for s in range(len(sentences)):
                find_flag = -1

                while sentences[s].find('_', find_flag + 1) != -1:
                    find_flag = sentences[s].find('_', find_flag + 1)
                    mask_indices.append([s, find_flag])
                    if dir_name != 'test':
                        sentences[s] = sentences[s].replace('_', option[ctr][answer[ctr]], 1)
                    ctr = ctr + 1
            if dir_name != 'test':
                for i in range(len(mask_indices)):
                    masked_sentences.append(sentences[mask_indices[i][0]][0:mask_indices[i][1]] +
                                            sentences[mask_indices[i][0]][mask_indices[i][1]:].replace(
                                                option[i][answer[i]], '[MASK]', 1))
                    encode_dict = tokenizer.encode_plus(masked_sentences[i], max_length=BATCH_SIZE,
                                                        padding='max_length',
                                                        truncation=True)

                    complete_ids.append(
                        tokenizer.encode_plus(sentences[mask_indices[i][0]], max_length=BATCH_SIZE,
                                              padding='max_length',
                                              truncation=True)['input_ids'])

                    input_id.append(encode_dict['input_ids'])
                    input_type.append(encode_dict['token_type_ids'])
                    input_mask.append(encode_dict['attention_mask'])
                # Loss case 3
                input_id = torch.LongTensor(input_id)
                complete_ids = torch.LongTensor(complete_ids)
                complete_ids = complete_ids.masked_fill(input_id != 103, -100)
            else:
                for i in range(len(mask_indices)):
                    masked_sentences.append(sentences[mask_indices[i][0]][0:mask_indices[i][1]] +
                                            sentences[mask_indices[i][0]][mask_indices[i][1]:].replace(
                                                '_', '[MASK]', 1))
                    encode_dict = tokenizer.encode_plus(masked_sentences[i], max_length=40, padding='max_length',
                                                        truncation=True)
                    input_id.append(encode_dict['input_ids'])
                    input_type.append(encode_dict['token_type_ids'])
                    input_mask.append(encode_dict['attention_mask'])

                # Loss case 3
                input_id = torch.LongTensor(input_id)
                complete_ids = torch.LongTensor(complete_ids)

            input_ids.append(input_id.to(DEVICE))
            input_types.append(torch.LongTensor(input_type).to(DEVICE))
            input_masks.append(torch.LongTensor(input_mask).to(DEVICE))
            all_complete_ids.append(complete_ids.to(DEVICE))
            all_masked_indices.append(
                torch.nonzero(input_id == BLANK_ID)[:, 1].to(DEVICE))
    return input_ids, input_types, input_masks, options, answers, all_masked_indices.unsqueeze(
        0), all_complete_ids


def pack_loaders(train, dev, test):
    print("Packing train...")
    train_loaders, dev_loaders, test_loaders = [], [], []
    for i in tqdm(range(len(train[0]))):
        train_article = TensorDataset(train[0][i], train[1][i], train[2][i], train[3][i].unsqueeze(0), train[4][i].T,
                                      train[5][i])
        train_sampler = SequentialSampler(train_article)
        train_loader = DataLoader(train_article, sampler=train_sampler, batch_size=BATCH_SIZE)
        train_loaders.append(train_loader)

    print("Packing dev...")
    for i in tqdm(range(len(dev[0]))):
        try:
            dev_article = TensorDataset(dev[0][i], dev[1][i], dev[2][i], dev[3][i].unsqueeze(0), dev[4][i].T,
                                        dev[5][i])
        except AssertionError:
            print(i)
        dev_sampler = SequentialSampler(dev_article)
        dev_loader = DataLoader(dev_article, sampler=dev_sampler, batch_size=BATCH_SIZE)
        dev_loaders.append(dev_loader)

    print("Packing test...")
    for i in tqdm(range(len(test[0]))):
        try:
            test_article = TensorDataset(test[0][i], test[1][i], test[2][i], test[3][i].unsqueeze(0))
        except AssertionError:
            print(i)
        test_sampler = SequentialSampler(test_article)
        test_loader = DataLoader(test_article, sampler=test_sampler, batch_size=BATCH_SIZE)
        test_loaders.append(test_loader)
    joblib.dump(train_loaders, os.path.join(loader_path, 'train_loaders'))
    joblib.dump(dev_loaders, os.path.join(loader_path, 'dev_loaders'))
    joblib.dump(test_loaders, os.path.join(loader_path, 'test_loaders'))
    return train_loaders, dev_loaders, test_loaders


def get_saved_loaders(loader_path):
    print('Loading data...')
    return joblib.load(os.path.join(loader_path, 'train_loaders')), \
           joblib.load(os.path.join(loader_path, 'dev_loaders')), \
           joblib.load(os.path.join(loader_path, 'test_loaders'))


def eval(eval_loaders, model=None):
    model.eval()
    if model is None:
        model = torch.load(MODEL_FILE_NAME)
    opt_acc_sum = 0
    acc = 0
    with torch.no_grad():
        for i, loader in tqdm(enumerate(eval_loaders)):
            for ids, types, masks, options, answers, mask_ids in loader:
                ids, types, masks, options, answers, mask_ids = ids.to(DEVICE), types.to(DEVICE), masks.to(
                    DEVICE), options.to(DEVICE), answers.to(DEVICE), mask_ids.unsqueeze(1).to(DEVICE)
                output, option_opts, loss = model(ids, masks, types, mask_ids, options, answers)
                opt_acc = (option_opts == answers).sum().float() / option_opts.shape[0]
                opt_acc_sum += opt_acc.cpu()
                acc += 1
        return opt_acc_sum / acc


def train(model, train_loaders, eval_loaders, lr=1e-4):
    loss_ls = []
    acc_ls = []
    optimizer = transformers.AdamW(params=model.parameters(), lr=lr)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 0,
                                                             EPOCHS * len(train_loaders))
    # criterion = nn.CrossEntropyLoss()
    print(len(train_loaders))

    # sanity check
    # loader = next(iter(train_loaders))

    for epoch in range(EPOCHS):
        loss_sum = 0
        opt_acc_sum = 0
        acc = 0
        highest_dev_acc = 0
        for i, loader in tqdm(enumerate(train_loaders)):
            for ids, types, masks, options, answers, mask_ids in loader:
                ids, types, masks, options, answers, mask_ids = ids.to(DEVICE), types.to(
                    DEVICE), masks.to(
                    DEVICE), options.to(DEVICE), answers.to(DEVICE), mask_ids.unsqueeze(1).to(DEVICE)
                model.train()
                optimizer.zero_grad()
                output, option_opts, loss = model(ids, masks, types, mask_ids, options, answers)

                # print(opt_acc)

                # loss = criterion(vals_output, answers.squeeze(1))
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_sum += loss.item()
                opt_acc_sum += (option_opts == answers).sum().float() / option_opts.shape[0]
                acc += 1
                # print(scheduler.get_lr())
            if (i + 1) % (len(train_loaders) // 5) == 0:
                eval_acc = eval(eval_loaders, model)
                print(
                    "Epoch {}, no.{}, loss {}, train accuracy {}, dev accuracy {}, lr {}".format(epoch, i + 1,
                                                                                                 loss_sum / acc,
                                                                                                 opt_acc_sum / acc,
                                                                                                 eval_acc,
                                                                                                 scheduler.get_last_lr()))
                loss_ls.append(loss_sum / acc)
                acc_ls.append(opt_acc_sum / acc)
                acc = 0
                loss_sum = 0
                opt_acc_sum = 0

                if eval_acc > highest_dev_acc:
                    highest_dev_acc = eval_acc
                    torch.save(model, MODEL_FILE_NAME + str(eval_acc))
                    print("Model saved!")
                # model.summary_net()
    return loss_ls, acc_ls


def summary_model(model_path):
    read_model = torch.load(model_path)
    # summary(model=read_model, input_size=[[20, 40], [20, 40], [20, 40], [20, 40]],
    #         batch_size=BATCH_SIZE)
    read_model.summary_net()


def get_dataset(path, dataset_type='Dataset', read_from_mem=False):
    if dataset_type == 'Dataset':
        if read_from_mem:
            return joblib.load(os.path.join(path, 'train_dataset')), \
                   joblib.load(os.path.join(path, 'eval_dataset')),
        else:
            train_dataset, eval_dataset = ClozeDataset('train'), ClozeDataset('dev')
            joblib.dump(train_dataset, os.path.join(path, 'train_dataset'))
            joblib.dump(eval_dataset, os.path.join(path, 'eval_dataset'))
            return train_dataset, eval_dataset
    elif dataset_type == 'IterableDataset':
        if read_from_mem:
            return joblib.load(os.path.join(path, 'train_iter_dataset')), \
                   joblib.load(os.path.join(path, 'eval_iter_dataset')),
        else:
            train_dataset, eval_dataset = ClozeDataset('train'), ClozeDataset('dev')
            joblib.dump(train_dataset, os.path.join(path, 'train_iter_dataset'))
            joblib.dump(eval_dataset, os.path.join(path, 'eval_iter_dataset'))
            return train_dataset, eval_dataset
    else:
        raise ValueError(f"Wrong dataset type {dataset_type}.")
