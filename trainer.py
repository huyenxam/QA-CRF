from metrics.evaluate import *
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BiaffineNER
from tqdm import trange
import os
from tqdm import tqdm

def get_pred_entity(pred):
    start = 0
    end = 0
    try:
        idx1 = pred.index(1)
        idx2 = 0
    except:
        idx1 = 0
        try:
            idx2 = pred.index(2)
        except:
            idx2 = 0
    if idx1 != 0:
        start = idx1
        end = idx1
        for i in range(idx1 + 1, len(pred)):
            if pred[i] != 2:
                break
            else:
                end = i
    elif idx2 != 0:
        start = idx2
        end = idx2
        for i in range(idx2 + 1, len(pred)):
            if pred[i] != 2:
                break
            else:
                end = i

    return [start, end]



class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.save_folder = args.save_folder
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        self.model = BiaffineNER(args=args)
        self.model.to(self.device)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.best_score = 0


    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(dataset=self.train_dataset, sampler=train_sampler,
                                      batch_size=self.args.batch_size, num_workers=16)

        total_steps = len(train_dataloader) * self.args.num_epochs
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )

        for epoch in trange(self.args.num_epochs):
            train_loss = 0
            print('EPOCH:', epoch)
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'first_subword': batch[2],
                          'char_ids': batch[4],
                          'labels': batch[-1]
                         }

                loss = self.model(**inputs)
                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()

                # norm gradient
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                max_norm=self.args.max_grad_norm)

                # optimizer.zero_grad() 
                
                optimizer.step()
                # update learning rate
                scheduler.step()
            print('train loss:', train_loss / len(train_dataloader))
            self.eval('dev')

    def eval(self, mode):
        if mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'test':
            dataset = self.test_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset=dataset, sampler=eval_sampler, batch_size=self.args.batch_size,
                                     num_workers=16)

        self.model.eval()

        eval_loss = 0
        labels = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'first_subword': batch[2],
                      'char_ids': batch[4],
                     }

            seq_length = batch[-3]
            with torch.no_grad():
                scores, outputs = self.model(**inputs)
                
                for i in range(len(outputs)):
                    true_len = seq_length[i]
                    score = scores[i][:true_len]
                    pred = outputs[i][:true_len]

                    label_pre = get_pred_entity(pred=pred)
                    labels.append(label_pre)


            # eval_loss += loss.item()

        exact_match, f1 = evaluate(labels, mode)

        print()
        print(exact_match)
        print(f1)

        if f1 > self.best_score:
            self.save_model()
            self.best_score = f1

    def save_model(self):
        checkpoint = {
                      'epoch': self.args.num_epochs,  
                      'model': self.model,
                      'state_dict': self.model.state_dict(),
                      }
        path = os.path.join(self.save_folder, 'checkpoint.pth')
        torch.save(checkpoint, path)
        torch.save(self.args, os.path.join(self.args.save_folder, 'training_args.bin'))

    def load_model(self):
        path = os.path.join(self.save_folder, 'checkpoint.pth')
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])




