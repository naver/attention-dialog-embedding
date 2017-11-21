import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from settings import *

def model_output(args, input_variable_list, encoder, attn_sum, ffnn, max_context, max_length, encoder_emhn=None, feature=None):
    sem_list = Variable(torch.zeros(max_context, encoder.input_size))
    sem_list = sem_list.cuda() if use_cuda else sem_list
    last_hidden_list = Variable(torch.zeros(max_context, encoder.hidden_size))
    last_hidden_list = last_hidden_list.cuda() if use_cuda else last_hidden_list
    sem_hidden_list = None
    if args.sem == "emhn":
        sem_hidden_list = Variable(torch.zeros(max_context, encoder_emhn.hidden_size))
        sem_hidden_list = sem_hidden_list.cuda() if use_cuda else sem_hidden_list
    
    if type(input_variable_list[0]) == str:
        input_variable_list = [variableFromSentence_glove(input_dial, var) for var in input_variable_list]
        input_variable_list = [var.cuda() for var in input_variable_list]
    encoder_hidden = encoder.initHidden()
    dial_len = len(input_variable_list)
    
    for idx, input_variable in enumerate(input_variable_list):
        try:
            input_length = input_variable.size()[0]
        except:
            print(input_variable_list)
            raise ValueError("There's no input word in the data")
        encoder_embeds = Variable(torch.zeros(max_length, encoder.input_size))
        encoder_embeds = encoder_embeds.cuda() if use_cuda else encoder_embeds
        encoder_hiddens = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_hiddens = encoder_hiddens.cuda() if use_cuda else encoder_hiddens

        for wordi in range(input_length):
            try:
                # input = input_variable[wordi].cpu().data
                encoder_output, encoder_hidden, encoder_embed = encoder(input_variable[wordi].data, encoder_hidden)
                encoder_embeds[wordi] = encoder_embed[0][0]
            except:
                print("encoder failed")
        # Last hidden state as sentence embedding
        last_hidden_list[21 - dial_len + idx] = encoder_hidden[ENC_LAYER - 1][0]
        # Average GloVe as sentence embedding
        sem = make_context(encoder_embeds, input_length)
        sem_list[21 - dial_len + idx] = sem
        
    # sem_list = sem_list[0:len(input_variable_list)]
    if use_cuda:
        sem = sem.cuda()
        encoder_hidden = encoder_hidden.cuda()

    if args.sem == "emhn":  
        hidden_sem = encoder_emhn.initHidden()
        for i in range(dial_len):
            output, hidden_sem = encoder_emhn(sem_list[21-dial_len+i].view(1,1,-1), hidden_sem)
            sem_hidden_list[21-dial_len+i] = hidden_sem
    if args.not_use_attention:
        if attn_sum.mode == "emh":
            concat_sem_hidden = torch.cat((sem_list, last_hidden_list), 1)
            last_sem = torch.cat((sem.view(1,-1), encoder_hidden[ENC_LAYER - 1]),1)
        elif attn_sum.mode == "em":
            concat_sem_hidden = sem_list
            last_sem = sem.view(1,-1)
        elif attn_sum.mode == "h":
            concat_sem_hidden = last_hidden_list
            last_sem = encoder_hidden[ENC_LAYER-1]
        elif attn_sum.mode =="emhn":
            concat_sem_hidden = torch.cat((sem_list, last_hidden_list), 1)
            concat_sem_hidden = torch.cat((concat_sem_hidden, sem_hidden_list), 1) 
            last_sem = torch.cat((sem.view(1,-1), encoder_hidden[ENC_LAYER - 1]),1)
            last_sem = torch.cat((last_sem, hidden_sem.view(1,-1)),1)

        hidden_broad = last_sem[0].expand_as(concat_sem_hidden)
        concat_context_hidden = torch.cat((concat_sem_hidden, hidden_broad), 1)
        context_vector, attn_weights = torch.mean(concat_context_hidden, 0), None
    else:
        context_vector, attn_weights = attn_sum(sem, sem_list, last_hidden_list, encoder_hidden, sem_hidden_list)
    context_vector = context_vector
    if feature is not None:
        feature = torch.FloatTensor(feature)
        feature = Variable(feature)
        feature = feature.cuda() if use_cuda else feature
        context_vector = torch.cat((context_vector, feature), 0)
    output = ffnn(context_vector)
    return output, attn_weights, context_vector

def train(args, input_variable_list, target_variable, encoder, attn_sum, ffnn, encoder_optim, attn_optim, ffnn_optim, criterion, max_length=MAX_LENGTH, max_context=MAX_CONTEXT, encoder_emhn=None, feature=None):
    encoder_optim.zero_grad()
    if not args.not_use_attention:
        attn_optim.zero_grad()
    ffnn_optim.zero_grad()
    loss = 0
    output, attn_weights, context_vector = model_output(args, input_variable_list, encoder, attn_sum, ffnn, max_context, max_length, encoder_emhn, feature)
    if args.criterion == "mse":
        loss = criterion(output, target_variable)
        # print(output)
        # print(target_variable)
    elif args.criterion == "multisoft":
        output_labels = []
        target_labels = []
        #output_label = 
        #target_label = 
        loss = criterion(output_labels, target_labels)
    loss.backward()

    encoder_optim.step()
    if not args.not_use_attention:
        attn_optim.step()
    ffnn_optim.step()

    return loss.data[0], context_vector, attn_weights


def trainIters(args, data, encoder, attnsum, ffnn, encoder_optim, attn_optim, ffnn_optim, criterion, print_every=100, plot_every=100, cur_epoch=0, epoch=1, encoder_emhn=None, features=None):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    tot_loss = 0
    iter = 1
    len_data = len(data)

    for idx, pair in enumerate(data):
        input_variable_list = pair[1]
        target_variable = pair[0][1]
        if use_cuda:
            input_variable_list = [var.cuda() for var in input_variable_list]
            target_variable = target_variable.cuda()
        if features is None:
            feature = None
        else:
            feature = features[idx]
        loss, context_vector, attn_weights = train(args, input_variable_list, target_variable,
                                                 encoder, attnsum, ffnn, encoder_optim, attn_optim, ffnn_optim, criterion, encoder_emhn=encoder_emhn, feature=feature)
        # train_ffnn(input_variable_list, target_variable, encoder, attnsum, ffnn, encoder_optim, attn_optim, ffnn_optim, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        tot_loss += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, (iter + cur_epoch * len_data) /
                                                   (epoch * len_data)), iter, iter / len_data * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        iter += 1
    tot_loss /= len_data
    # showPlot(plot_losses)
    return plot_losses, tot_loss, (context_vector, attn_weights)
