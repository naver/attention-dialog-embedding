import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from settings import *


class Encoder_glove(nn.Module):
    def __init__(self, glove_voc, glove_vec, hidden_size=HIDDEN_SIZE, n_layers=1):
        super(Encoder_glove, self).__init__()
        self.n_layers = n_layers
        self.input_size = glove_vec.size()[1]
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, ENC_LAYER)
        self.glove_vec = glove_vec
        self.glove_voc = glove_voc

    def forward(self, input, hidden):
        embedded = self.glove_vec[input.cpu()].view(1, 1, -1)
        output = Variable(embedded)
        output = output.cuda() if use_cuda else output
        output, hidden = self.gru(output, hidden)
        return output, hidden, embedded

    def initHidden(self):
        result = Variable(torch.zeros(ENC_LAYER, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Encoder_emhn(nn.Module):
    def __init__(self, input_size, output_size, n_layers=1):
        super(Encoder_emhn, self).__init__()
        self.hidden_size = output_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, self.hidden_size, self.n_layers)
    def forward(self, input, hidden):
        # output = output.cuda() if use_cuda else output
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnSum(nn.Module):
    def __init__(self, hidden_size, sem_mode="emh", dropout_p=0.1):
        super(AttnSum, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
        self.hs = hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.mode = sem_mode

    def forward(self, sem, sem_list, hidden_list, final_hidden, sem_hidden=None):
        sem_list = self.dropout(sem_list)
        
        #hidden_broad = final_hidden[ENC_LAYER - 1].expand_as(hidden_list)
        
        if self.mode == "emh":
            concat_sem_hidden = torch.cat((sem_list, hidden_list), 1)
            last_sem = torch.cat((sem, final_hidden[ENC_LAYER - 1][0]),0)
        elif self.mode == "em":
            concat_sem_hidden = sem_list
            last_sem = sem
        elif self.mode == "h":
            concat_sem_hidden = hidden_list
            last_sem = final_hidden[ENC_LAYER - 1][0]
        elif self.mode =="emhn":
            concat_sem_hidden = torch.cat((sem_list, hidden_list), 1)
            concat_sem_hidden = torch.cat((concat_sem_hidden, sem_hidden), 1) 
            last_sem = torch.cat((sem, final_hidden[ENC_LAYER - 1][0]),0)
            last_sem = torch.cat((last_sem, sem_hidden[-1]),0)
        hidden_broad = last_sem.expand_as(concat_sem_hidden)
        concat_context_hidden = torch.cat((concat_sem_hidden, hidden_broad), 1)
        # concat_context_hidden = concat_context_hidden.cpu()
        attn_weights = func.softmax(self.attn(concat_context_hidden).view(1, -1))
        # attn_weights = attn_weights.cuda() if use_cuda else attn_weights
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), concat_sem_hidden.unsqueeze(0))
        output = torch.cat((attn_applied[0][0], last_sem),0)
        # if self.mode == "emh":
        #     output = torch.cat((sem.view(1, -1), final_hidden.view(1,-1), attn_applied[0]), 1)
        # elif self.mode == "em":
        #     output = torch.cat((sem.view(1, -1), attn_applied[0]), 1)
        # elif self.mode == "h":
        #     output = torch.cat((final_hidden.view(1, -1), attn_applied[0]), 1)
        # elif self.mode == "emhn":
        #     output = torch.cat((sem.view(1, -1), final_hidden.view(1,-1), sem_hidden.view(1,-1), attn_applied[0]), 1)
        output = output.cuda() if use_cuda else output
        return output, attn_weights


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if hidden_size > 0:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
        else:
            self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        if self.hidden_size > 0:
            out = self.fc2(out)
        return out
