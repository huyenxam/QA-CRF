import torch
import json
import numpy as np
from torch.utils.data import Dataset


class InputSample(object):
    def __init__(self, path, label_set_path=None, max_char_len=None, max_seq_length=None):
        self.max_char_len = max_char_len            # Độ dài tối đa đầu vào của CharCNN
        self.max_seq_length = max_seq_length        # Độ dài tối của context
        with open(label_set_path, 'r', encoding='utf8') as f:  
            self.label_set = f.read().splitlines()
        self.label_2int = {w: i for i, w in enumerate(self.label_set)}     

        self.list_sample = []                       # Danh sách các mẫu
        with open(path, 'r', encoding='utf8') as f: # Đọc file data
            self.list_sample = json.load(f)
        # self.list_sample = self.list_sample[:10]

    def get_character(self, word, max_char_len):
        word_seq = []
        for j in range(max_char_len):
            try:
                char = word[j]
            except:
                char = 'PAD'
            word_seq.append(char)
        return word_seq

        
    def get_sample(self):
        l_sample = []
        for sample in self.list_sample:
            qa_dict = {}   
            context = sample['context'].split(' ')
            question = sample['question'].split(' ')
            
            max_seq = self.max_seq_length - len(question) - 3       
            if len(context) > max_seq:
                context = context[:max_seq]

            sent = question + context    
            char_seq = []
            for word in sent:
                character = self.get_character(word, self.max_char_len)
                char_seq.append(character)

            label = sample['answers'][0]
            start =  label[1] 
            end = label[2] 

            labels_idx = [self.label_2int['CLS']]
            labels_idx += [self.label_2int['B-Question'] if i == 0 else self.label_2int['I-Question'] for i in range(len(question))]
            labels_idx += [self.label_2int['SEP']]
            labels_idx += [self.label_2int['I-Answer'] if (i >= start and i <= end) else self.label_2int['O'] for i in range(len(context))]
            labels_idx[len(question) + 2] = self.label_2int['B-Answer']

            qa_dict['context'] = context
            qa_dict['question'] = question
            qa_dict['label_idx'] = labels_idx
            qa_dict['char_sequence'] = char_seq
            qa_dict['label'] = sample['label']
            l_sample.append(qa_dict)

        return l_sample


class MyDataSet(Dataset):

    def __init__(self, path, char_vocab_path, label_set_path,
                 max_char_len, tokenizer, max_seq_length):

        self.samples = InputSample(path=path, label_set_path=label_set_path, max_char_len=max_char_len, max_seq_length=max_seq_length).get_sample()
        self.tokenizer = tokenizer                             
        self.max_seq_length = max_seq_length           
        self.max_char_len = max_char_len               

        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)

    def preprocess(self, tokenizer, context, question, max_seq_length, mask_padding_with_zero=True):
        firstSWindices = [0]
        input_ids = [tokenizer.cls_token_id]                    
        firstSWindices.append(len(input_ids))

        for w in question:
            word_token = tokenizer.encode(w)                   
            input_ids += word_token[1: (len(word_token) - 1)]  
            firstSWindices.append(len(input_ids))              
        
        input_ids.append(tokenizer.sep_token_id)             
        firstSWindices.append(len(input_ids))

        for w in context:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            if len(input_ids) >= max_seq_length:                
              firstSWindices.append(0)
            else:
              firstSWindices.append(len(input_ids))
              
        firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
        input_ids.append(tokenizer.sep_token_id)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        if len(input_ids) > max_seq_length:             
            input_ids = input_ids[:max_seq_length]
            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))
            firstSWindices = firstSWindices[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (max_seq_length - len(input_ids))
            input_ids = (
                    input_ids
                    + [
                        tokenizer.pad_token_id,
                    ]
                    * (max_seq_length - len(input_ids))
            )

            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(firstSWindices)

    def character2id(self, character_sentence, max_seq_length):
        char_ids = []
        for word in character_sentence:                
            word_char_ids = []
            for char in word:                          
                if char not in self.char_vocab:    
                    word_char_ids.append(self.char_vocab['UNK'])
                else:                                  
                    word_char_ids.append(self.char_vocab[char])
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_length:            
            char_ids += [[self.char_vocab['PAD']] * self.max_char_len] * (max_seq_length - len(char_ids))
        else:
            char_ids = char_ids[:max_seq_length]
        return torch.tensor(char_ids)

    def __getitem__(self, index):

        sample = self.samples[index]
        context = sample['context']
        question = sample['question']
        char_seq = sample['char_sequence']
        seq_length = len(question) + len(context) + 2        
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, context, question, self.max_seq_length)
        if len(label) > self.max_seq_length:
            label = label[:self.max_seq_length]
        label = torch.Tensor(sample['label_idx']).to(torch.int32)

        char_ids = self.character2id(char_seq, max_seq_length=self.max_seq_length)
        if seq_length > self.max_seq_length:
          seq_length = self.max_seq_length

        return input_ids, attention_mask, firstSWindices, torch.tensor([seq_length]), char_ids, label.long()

    def __len__(self):
        return len(self.samples)
