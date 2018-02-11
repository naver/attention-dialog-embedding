import torch
import random
import argparse
from utils import *
from models import *
from trainer import *
from evaluater import *
from parser import args
from torch import optim
from settings import *
from eval_res import print_eval, print_eval_short
from torch.optim.lr_scheduler import *


def train_all(args, train_vars, test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler, encoder_sem=None):
    epoch_losses = []
    for ep in range(args.ae):
        if not args.sem =='emhn':
            plot_losses, tot_loss, etc = trainIters(args.sem, train_vars, encoder, attn, ffnn, encoder_optim,
                                                attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=ep, epoch=args.ae)
        else:
            plot_losses, tot_loss, etc = trainIters_sem(args.sem, train_vars, encoder, encoder_sem, attn, ffnn, encoder_optim, encoder_sem_optim,
                                                attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=ep, epoch=args.ae)
        # Evaluate
        if args.sem == "emhn":
            model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, args.sem, encoder, attn, ffnn, criterion, input_dial, encoder_sem=encoder_sem)
        else:
            model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, args.sem, encoder, attn, ffnn, criterion, input_dial)

        model_target = [list(x[0][1].view(-1).data.cpu().numpy()) for x in test_vars]
        eval_input = [model_output[x] + model_target[x] for x in range(len(model_target))]
        epoch_losses.append(model_loss)        
        if not args.nodecay:
            attn_scheduler.step(model_loss)
            ffnn_scheduler.step(model_loss)
        if model_loss == min(epoch_losses):
            best_loss = model_loss

    return (best_loss, epoch_losses)

def train_data(args, train_vars, test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler, encoder_sem=None):
    epoch_accs = []
    epoch_losses = []
    model_target = [list(x[0][1].view(-1).data.cpu().numpy()) for x in test_vars]
    if args.sem == "emhn":
        model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, args.sem, encoder, attn, ffnn, criterion, input_dial, encoder_sem=encoder_sem)
    else:
        model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, args.sem, encoder, attn, ffnn, criterion, input_dial)

    for ep in range(args.de):
        if not args.sem =='emhn':
            plot_losses, tot_loss, etc = trainIters(args.sem, train_vars, encoder, attn, ffnn, encoder_optim,
                                                attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=ep, epoch=args.de)
        else:
            plot_losses, tot_loss, etc = trainIters_sem(args.sem, train_vars, encoder, encoder_sem, attn, ffnn, encoder_optim, encoder_sem_optim,
                                                attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=ep, epoch=args.de)
        # Evaluate
        if args.sem == "emhn":
            model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, args.sem, encoder, attn, ffnn, criterion, input_dial, encoder_sem=encoder_sem)
        else:
            model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, args.sem, encoder, attn, ffnn, criterion, input_dial)

        eval_input = [model_output[x] + model_target[x] for x in range(len(model_target))]
        acc, fval = print_eval_short(eval_input, args.th)
        epoch_accs.append(acc)
        epoch_losses.append(model_loss)        
        if acc ==max(epoch_accs):
            best_acc = acc
            best_fval = fval
        if not args.nodecay:
            attn_scheduler.step(model_loss)
            ffnn_scheduler.step(model_loss)
        if model_loss == min(epoch_losses):
            best_loss = model_loss

    return (best_acc, best_fval, best_loss, epoch_losses)

if __name__ == "__main__":
    # load data
    dial_list, input_dial, output_class, pairs = prepareData('./data/dial_list_all_unk.txt', './data/tgt_list_all.txt')
    i_dial, i_pairs = dial_list[:100], pairs[:100]
    t_dial, t_pairs = dial_list[100:200], pairs[100:200]
    c_dial, c_pairs = dial_list[200:315], pairs[200:315]
    y_dial, y_pairs = dial_list[315:], pairs[315:]

    if args.glove == 50:
        load_glove("./data/glove.twitter.27B.50d_special.txt", input_dial, args.load_glove)
    elif args.glove == 100:
        load_glove("./data/glove.twitter.27B.100d.special.txt", input_dial, args.load_glove)

    i_train_vars, i_test_vars = split_data(i_pairs, 0.1, i_dial, input_dial)
    t_train_vars, t_test_vars = split_data(t_pairs, 0.1, t_dial, input_dial)
    c_train_vars, c_test_vars = split_data(c_pairs, 0.1, c_dial, input_dial)
    y_train_vars, y_test_vars = split_data(y_pairs, 0.1, y_dial, input_dial)

    train_vars = i_train_vars+t_train_vars+c_train_vars+y_train_vars
    test_vars = i_test_vars+t_test_vars+c_test_vars+y_test_vars

    # train_vars = pickle.load(open('./data/train_vars_all_1', 'rb'))
    # test_vars = pickle.load(open('./data/test_vars_all_1', 'rb'))

    # vars_dict= [hidden_size, learning_rate, sem_method, decay_rate, decay_patience, dropout_p, l2_norm, glove_dim]
    if args.test:
        train_vars = random.sample(train_vars, 200)
        test_vars = random.sample(test_vars, 50)
    if args.toy:
        train_vars = random.sample(train_vars, 1000)
        test_vars = random.sample(test_vars, 100)

    random.shuffle(train_vars)
    random.shuffle(test_vars)
    encoder = EncoderRNN_glove(input_dial.glove_voc, input_dial.glove_vec, n_layers=ENC_LAYER)

    if args.sem == 'emh':
        attn = AttnSum(input_dial.glove_vec.size()[1] +args.hs*2, args.sem, dropout_p=args.dp)
        ffnn = FFNN(input_dial.glove_vec.size()[1] *2+args.hs, args.ff, output_class.n_class)
    elif args.sem == 'h': 
        attn = AttnSum(args.hs*2, args.sem, dropout_p=args.dp)
        ffnn = FFNN(input_dial.glove_vec.size()[1]+args.hs, args.ff, output_class.n_class)
    elif args.sem =='em':
        attn = AttnSum(input_dial.glove_vec.size()[1] +args.hs*1, args.sem, dropout_p=args.dp)
        ffnn = FFNN(input_dial.glove_vec.size()[1] *2, args.ff, output_class.n_class)
    elif args.sem == 'emhn':
        encoder_sem = Encoder_sem(input_dial.glove_vec.size()[1], input_dial.glove_vec.size()[1], n_layers=1)
        encoder_sem_optim = optim.Adam(encoder_sem.parameters(), lr=args.lr, weight_decay=args.l2)
        encoder_sem = encoder_sem.cuda() if use_cuda else encoder_sem
        attn = AttnSum(input_dial.glove_vec.size()[1] *2 + args.hs*2, args.sem, dropout_p=args.dp)
        ffnn = FFNN(input_dial.glove_vec.size()[1] * 2+encoder_sem.hidden_size + args.hs, args.ff, output_class.n_class)

    if use_cuda:
        encoder = encoder.cuda()
        attn = attn.cuda()
        ffnn = ffnn.cuda()

    encoder_optim = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
    attn_optim = optim.Adam(attn.parameters(), lr=args.lr, weight_decay=args.l2)
    ffnn_optim = optim.Adam(ffnn.parameters(), lr=args.lr, weight_decay=args.l2)
    
    if not args.nodecay:
        attn_scheduler = ReduceLROnPlateau(attn_optim, factor=args.dr, patience=args.pat, verbose=True, min_lr=0.00001)
        ffnn_scheduler = ReduceLROnPlateau(ffnn_optim, factor=args.dr, patience=args.pat, verbose=True, min_lr=0.00001)
    criterion = nn.MSELoss()

    # Train

    model_target = [list(x[0][1].view(-1).data.cpu().numpy()) for x in test_vars]
    if args.sem == 'emhn':
        train_all(args, train_vars, test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler, encoder_sem)
        if args.data == "i":
            train_data(args, i_train_vars, i_test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler, encoder_sem)
        if args.data == "t":
            train_data(args, t_train_vars, t_test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler, encoder_sem)
        if args.data == "c":
            train_data(args, c_train_vars, c_test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler, encoder_sem)
        if args.data == "y":
            train_data(args, y_train_vars, y_test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler, encoder_sem)
    else:  
        train_all(args, train_vars, test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler)
        if args.data == "i":
            train_data(args, i_train_vars, i_test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler)
        if args.data == "t":
            train_data(args, t_train_vars, t_test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler)
        if args.data == "c":
            train_data(args, c_train_vars, c_test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler)
        if args.data == "y":
            train_data(args, y_train_vars, y_test_vars, input_dial, output_class, pairs, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, attn_scheduler, ffnn_scheduler)
    
    # timestr = time.strftime("%m%d_%H%M%S")
    # with open('./loss_figures/' + timestr + '_grid_' + '.txt', 'w') as file:
    #     file.write(str(model_zip))
