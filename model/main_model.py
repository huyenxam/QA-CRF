from torch import nn
from model.layer import WordRep
from transformers import AutoConfig
from torchcrf import CRF


class BiaffineNER(nn.Module):
    def __init__(self, args):
        super(BiaffineNER, self).__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.num_labels = args.num_labels
        self.lstm_input_size = args.num_layer_bert * config.hidden_size
        if args.use_char:
            self.lstm_input_size = self.lstm_input_size + 2 * args.char_hidden_dim

        self.word_rep = WordRep(args)
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim // 2,
                              num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.classifier = nn.Linear(args.hidden_dim , self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)


    def forward(self, input_ids=None, char_ids=None,  first_subword=None, attention_mask=None, labels=None):

        x = self.word_rep(input_ids=input_ids, attention_mask=attention_mask,
                                      first_subword=first_subword,
                                      char_ids=char_ids)
        # x = [bs, max_sep, 768 + char_hidden_dim*2]
        x, _ = self.bilstm(x)
        # x = [bs, max_sep, hidden_dim]
        x = self.dropout(x)
        # x = [bs, max_sep, hidden_dim]
        scores = self.classifier(x)
        # x = [bs, max_seq, 2]
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(scores, labels, loss_mask)*(-1)
            return loss
        else:
            # return scores
            outputs = self.crf.decode(scores)
            return scores, outputs