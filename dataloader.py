import torch
import json
import numpy as np
from torch.utils.data import Dataset


class InputSample(object):
    def __init__(self, path, max_char_len=None, max_seq_length=None):
        self.max_char_len = max_char_len            # Độ dài tối đa đầu vào của CharCNN
        self.max_seq_length = max_seq_length        # Độ dài tối của context
        self.list_sample = []                       # Danh sách các mẫu
        with open(path, 'r', encoding='utf8') as f: # Đọc file data
            self.list_sample = json.load(f)

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

            label = sample['label'][0]
            entity = label[0]
            start =  label[1] + len(question) + 2
            end = label[2] + len(question) + 2
            if end >= self.max_seq_length:
                end = self.max_seq_length - 1

            if start >= self.max_seq_length - 1:
                start = 0
                end = 0
            
            qa_dict['context'] = context
            qa_dict['question'] = question
            qa_dict['label_idx'] = [entity, start, end]
            qa_dict['char_sequence'] = char_seq
            qa_dict['label'] = sample['label']
            l_sample.append(qa_dict)

        return l_sample


class MyDataSet(Dataset):

    def __init__(self, path, char_vocab_path, label_set_path,
                 max_char_len, tokenizer, max_seq_length):

        self.samples = InputSample(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length).get_sample()
        self.tokenizer = tokenizer                      # Chuyển văn bản thô thành số sử dụng tokenizer Phobert
                                                        # Example: "Chúng_tôi là những nghiên_cứu_viên ." -> tensor([[    0,   746,     8,    21, 46349,     5,     2]])                
        self.max_seq_length = max_seq_length            # Độ dài tối đa của context
        self.max_char_len = max_char_len                # Độ dài tối đã của char
        with open(label_set_path, 'r', encoding='utf8') as f:   # Đọc file từ điển nhãn
            self.label_set = f.read().splitlines()

        with open(char_vocab_path, 'r', encoding='utf-8') as f: # Đọc file từ điển các ký tự
            self.char_vocab = json.load(f)
        self.label_2int = {w: i for i, w in enumerate(self.label_set)}      

    def preprocess(self, tokenizer, context, question, max_seq_length, mask_padding_with_zero=True):
        firstSWindices = [0]
        input_ids = [tokenizer.cls_token_id]                    # Thêm [CLS] vào đầu câu
        firstSWindices.append(len(input_ids))

        for w in question:
            word_token = tokenizer.encode(w)                    # Chuyển các token thành số
            input_ids += word_token[1: (len(word_token) - 1)]   # Chỉ lấy token đầu tiên
                                                                # Example: seq = "Chúng tôi"
                                                                # tokenizer.encode("Chúng tôi") -> [0, 746, 2]
                                                                # Lấy token đầu tiên tại vị trí [1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))               # lưu lại vị trí token đã lấy 
        
        input_ids.append(tokenizer.sep_token_id)                # Thêm [SEP] và giữa question và context
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
        for word in character_sentence:                 # Lặp qua từng câu trong sentence
            word_char_ids = []
            for char in word:                           # Lặp qua từng ký tự trong câu
                if char not in self.char_vocab:         # Nếu ký tự không xuất hiện trong từ điển ký tự thì gán nó bằng 'UNK'
                    word_char_ids.append(self.char_vocab['UNK'])
                else:                                   # Nếu xuất hiện trong từ điển thì chuyển ký tự thành số 
                    word_char_ids.append(self.char_vocab[char])
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_length:              # Nếu độ dài câu nhỏ hơn max_seq_length thì thêm padding vào cuối
            char_ids += [[self.char_vocab['PAD']] * self.max_char_len] * (max_seq_length - len(char_ids))
        else:
            char_ids = char_ids[:max_seq_length]
        return torch.tensor(char_ids)

    def span_maxtrix_label(self, label):   
        start = int(label[1])
        end = int(label[2])
        en = label[0]
        if start > self.max_seq_length or end > self.max_seq_length:       
            start = 0
            end = 0
        try:
            entity = self.label_2int[en]
        except:
            print(label)
        
        label = torch.Tensor([entity if i >= start and i <= end else 0 for i in range(self.max_seq_length)])
        return label

    def __getitem__(self, index):

        sample = self.samples[index]
        context = sample['context']
        question = sample['question']
        char_seq = sample['char_sequence']
        seq_length = len(question) + len(context) + 2        
        label = sample['label_idx']
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, context, question, self.max_seq_length)

        char_ids = self.character2id(char_seq, max_seq_length=self.max_seq_length)
        if seq_length > self.max_seq_length:
          seq_length = self.max_seq_length
        label = self.span_maxtrix_label(label)

        return input_ids, attention_mask, firstSWindices, torch.tensor([seq_length]), char_ids, label.long()

    def __len__(self):
        return len(self.samples)
