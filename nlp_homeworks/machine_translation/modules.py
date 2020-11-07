#MODEULES.PY

import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x): # с tempreture=10, отвечает за гладкость
    e_x = torch.exp(x / 10)
    return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.tanh = nn.Tanh()

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)
        
    def forward(self, hidden, encoder_outputs):
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]
        
        encoder_len = encoder_outputs.shape[0]
        hidden = torch.tensor(hidden)

        # repeat hidden and concatenate it with encoder_outputs
        hiddens = hidden.expand(encoder_len * hidden.shape[0], -1, -1)
        attn_concat = torch.cat([hiddens, encoder_outputs], dim = 2)
        # calculate energy
        
        result_of_attn = self.attn(attn_concat)
        energy = self.tanh(result_of_attn)
        attn_weights = self.v(energy)

        # get attention, use softmax function which is defined, can change temperature
        attn_final = softmax(attn_weights)
        attn_final = attn_final.permute((1, 2, 0))
            
        return attn_final
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(input_size = emb_dim + dec_hid_dim,
                         hidden_size  = dec_hid_dim, 
                         num_layers = 1, dropout = dropout)
        
        self.out = nn.Linear(in_features = dec_hid_dim * 2 + emb_dim, out_features = output_dim) # linear layer to get next word
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        
        input = input.unsqueeze(0) # because only one word, no words sequence 
        embedded = self.dropout(self.embedding(input))
        
        # get weighted sum of encoder_outputs
        attention_v = self.attention(hidden, encoder_outputs)
        encoder_outputs = encoder_outputs.permute((1, 0, 2))
        weighted_sum = torch.bmm(attention_v, encoder_outputs)
        weighted_sum = weighted_sum.permute((1, 0, 2))

        # concatenate weighted sum and embedded, break through the GRU
        gru_concat = torch.cat([embedded, weighted_sum], dim = 2)
        
        # get predictions
        output, hidden = self.rnn(gru_concat, hidden)

        concat_w_last_layer = torch.cat([embedded, weighted_sum,
                                         output], dim = 2)
        
        concat_w_last_layer = concat_w_last_layer.squeeze(0)
        
        prediction = self.out(concat_w_last_layer)

        return prediction, hidden
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):

            output, hidden = self.decoder(input, hidden, enc_states)
            
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            # top1 or ground truth
            input = (trg[t] if teacher_force else top1)
        
        return outputs