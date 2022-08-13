from metrics.evaluate import *
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BiaffineNER
from tqdm import trange
import os
from tqdm import tqdm

# def get_pred_entity(cate_pred, span_scores,label_set, is_flat_ner= True):
#     top_span = []
#     for i in range(len(cate_pred)):
#         for j in range(i,len(cate_pred)):
#             if cate_pred[i][j]>0:
#                 tmp = (label_set[cate_pred[i][j].item()], i, j,span_scores[i][j].item())
#                 top_span.append(tmp)
#     top_span = sorted(top_span, reverse=True, key=lambda x: x[3])
    
#     if not top_span:
#         top_span = [('ANSWER', 0, 0)]

#     return top_span[0]

def get_pred_entity(score):
    top_span = []
    for i in range(len(score)):
        if score[i][1] > score[i][0]:
            sum_score = score[i][1]
            for j in range(i,len(score)):
                if score[j][1] > score[j][0]:
                    break
                sum_score += score[j][1]
            top_span.append(("ANSWER", i, j, sum_score))
    top_span = sorted(top_span, reverse=True, key=lambda x: x[3])
    if not top_span:
        top_span = [('ANSWER', 0, 0)]

    return top_span[0]



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
        self.label_set = train_dataset.label_set

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
                outputs = self.model(**inputs)
                
                # print(outputs)
                for i in range(len(outputs)):
                    predict = outputs[i][0]
                    score = outputs[i][1]

                    label_pre = get_pred_entity(predict=predict, score=score)
                    print(label_pre)
                    labels.append(label_pre)

        #         for i in range(len(output)):
        #             # out = output[i].max(dim=-1)
        #             true_len = seq_length[i]
        #             out = output[i][:true_len, :true_len]
                    
        #             input_tensor, cate_pred = out.max(dim=-1)
                    
        #             label_pre = get_pred_entity(cate_pred, input_tensor, self.label_set, True)
        #             outputs.append(label_pre)
        #             # labels.append(label1)
        #             # print(label_pre)

        #     eval_loss += loss.item()

        # exact_match, f1 = evaluate(outputs, mode)

        # print()
        # print(exact_match)
        # print(f1)

        # if f1 > self.best_score:
        #     self.save_model()
        #     self.best_score = f1

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