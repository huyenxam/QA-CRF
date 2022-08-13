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
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.classifier = nn.Linear(self.lstm_input_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)


    def forward(self, input_ids=None, char_ids=None,  first_subword=None, attention_mask=None, labels=None):

        x = self.word_rep(input_ids=input_ids, attention_mask=attention_mask,
                                      first_subword=first_subword,
                                      char_ids=char_ids)
        # x = [bs, max_sep, 768 + char_hidden_dim*2]
        x = self.dropout(x)
        # x = [bs, max_sep, 768 + char_hidden_dim*2]
        logits = self.classifier(x)
        # x = [bs, max_seq, 2]
        outputs = (logits, )
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask)*(-1)
            outputs = (loss,) + outputs
        return outputs
        