import torch
import random
import argparse
from utils import *
from models import *
from trainer import *
from evaluater import *
from torch import optim
from settings import *
from eval_res import print_eval, print_eval_short
from torch.optim.lr_scheduler import *
from parser import args

# load data
dial_list, input_dial, output_class, pairs = prepareData('./data/dial_list_all_unk.txt', './data/tgt_list_all.txt')
if args.movie or args.mix_movie:
    dial_list_movie, input_dial_movie, output_class_movie, pairs_movie = prepareData('./data/dial_list_movie_unk2.txt', './data/tgt_list_movie.txt', movie=True)

if args.glove == 50:
    load_glove("./data/glove.twitter.27B.50d_special.txt", input_dial, args.glove, args.load_glove)
    print("loaded glove-50 data")
elif args.glove == 100:
    load_glove("./data/glove.twitter.27B.100d.special.txt", input_dial, args.glove, args.load_glove)
    print("loaded glove-100 data")
if args.movie:
    load_glove("./data/glove.twitter.27B.100d.special.txt", input_dial_movie, args.glove, args.load_glove)

if args.no_feature:
    if args.ran:
    # split data to train, test data if option selcted as random
        train_vars, test_vars = split_data(pairs, 0.1, dial_list, input_dial)
        train_features, test_features = None, None
    else:
        train_vars = pickle.load(open('./data/train_vars_all_1', 'rb'))
        test_vars = pickle.load(open('./data/test_vars_all_1', 'rb'))
        train_features, test_features = None, None
else:
    with open('./data/features', 'rb') as file:
        features = pickle.load(file)
    if args.ran:
    # split data to train, test data if option selcted as random
        train_vars, test_vars, train_features, test_features= split_data_features(pairs, 0.1, dial_list, input_dial, features)
    else:
        train_vars = pickle.load(open('./data/train_vars_all_1', 'rb'))
        test_vars = pickle.load(open('./data/test_vars_all_1', 'rb'))
        train_features = pickle.load(open('./data/train_features_all_1', 'rb'))
        test_features = pickle.load(open('./data/test_features_all_1', 'rb'))
    train_vars = list(zip(train_vars, train_features))
    test_vars = list(zip(test_vars, test_features))
if args.movie or args.mix_movie:
    if os.path.isfile('./data/movie_train_vars'):
        with open('./data/movie_train_vars','rb') as file:
            train_vars_movie = pickle.load(file)
        print("loaded pre-saved movie_train_vars - len: "+str(len(train_vars_movie)))
    else:
        train_vars_movie = split_data_movie(pairs_movie, 0.1, dial_list_movie, input_dial_movie)
        with open('./data/movie_train_vars', 'wb') as file:
            pickle.dump(train_vars_movie, file)
        print("saved movie_train_vars")
    if os.path.isfile('./data/movie_train_false_vars'):
        with open('./data/movie_train_false_vars','rb') as file:
            train_false_vars_movie = pickle.load(file)
        print("loaded pre-saved movie_train_false_vars - len: "+str(len(train_false_vars_movie)))
    else:
        train_false_vars_movie = make_false_sample_movie(dial_list_movie, input_dial_movie)
        with open('./data/movie_train_false_vars', 'wb') as file:
            pickle.dump(train_false_vars_movie, file)
        print("saved movie_train_false_vars")
    train_false_vars_movie = random.sample(train_false_vars_movie, math.floor(len(train_vars_movie)*args.tf_ratio))
    train_vars_movie = train_vars_movie+train_false_vars_movie
    train_vars_movie = random.sample(train_vars_movie, math.floor(len(train_vars_movie)*args.movie_sample))
    random.shuffle(train_vars_movie)
    if args.mix_movie:
        train_vars = train_vars+train_vars_movie

# vars_dict= [hidden_size, learning_rate, sem_method, decay_rate, decay_patience, dropout_p, l2_norm, glove_dim]
if args.test:
    train_vars = random.sample(train_vars, 200)
    test_vars = random.sample(test_vars, 50)
if args.toy:
    train_vars = random.sample(train_vars, 900)
    test_vars = random.sample(test_vars, 100)
random.shuffle(train_vars)
random.shuffle(test_vars)
if not args.no_feature:
    train_zip = list(zip(*train_vars))
    train_vars = list(train_zip[0])
    train_features = list(train_zip[1])
    test_zip = list(zip(*test_vars))
    test_vars = list(test_zip[0])
    test_features = list(test_zip[1])

# load args
if args.resume:
    if os.path.isfile('./checkpoints/'+args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load('./checkpoints/'+args.resume)
        args = checkpoint['args']

def one_model(hyper, args, train_vars, test_vars, dial_list, input_dial, output_class, pairs, train_features=None, test_features=None):
    #if hyper['glove_dim'] == 50:
    #    load_glove("./data/glove.twitter.27B.50d_special.txt", input_dial, args.glove)
    #elif hyper['glove_dim'] == 100:
    #    load_glove("./data/glove.twitter.27B.100d.special.txt", input_dial, False)
    #print("loaded glove data")

    # define models, optimizers, loss criterion
    encoder = Encoder_glove(input_dial.glove_voc, input_dial.glove_vec, hyper['h'],n_layers=ENC_LAYER)

    if hyper['sem'] == 'emh':
        attn = AttnSum(input_dial.glove_vec.size()[1] +hyper['h']*2, hyper['sem'], dropout_p=hyper['dp'])
        ffnn = FFNN(input_dial.glove_vec.size()[1] *2+hyper['h'], hyper['ff'], output_class.n_class)
    elif hyper['sem'] == 'h': 
        attn = AttnSum(hyper['h']*2, hyper['sem'], dropout_p=hyper['dp'])
        ffnn = FFNN(input_dial.glove_vec.size()[1]+hyper['h'], hyper['ff'], output_class.n_class)
    elif hyper['sem'] =='em':
        attn = AttnSum(input_dial.glove_vec.size()[1] +hyper['h']*1, hyper['sem'], dropout_p=hyper['dp'])
        ffnn = FFNN(input_dial.glove_vec.size()[1] *2, hyper['ff'], output_class.n_class)
    elif hyper['sem'] == 'emhn':
        encoder_sem = Encoder_sem(input_dial.glove_vec.size()[1], input_dial.glove_vec.size()[1], n_layers=1)
        encoder_sem_optim = optim.Adam(encoder_sem.parameters(), lr=hyper['lr'], weight_decay=hyper['l2'])
        encoder_sem = encoder_sem.cuda() if use_cuda else encoder_sem
        attn = AttnSum(input_dial.glove_vec.size()[1] *2 + hyper['h']*2, hyper['sem'], dropout_p=hyper['dp'])
        ffnn = FFNN(input_dial.glove_vec.size()[1] * 2+encoder_sem.hidden_size + hyper['h'], hyper['ff'], output_class.n_class)
    if not args.no_feature:
        ffnn = FFNN(ffnn.input_size+15, ffnn.hidden_size, output_class.n_class)

    if use_cuda:
        encoder = encoder.cuda()
        attn = attn.cuda()
        ffnn = ffnn.cuda()

    encoder_optim = optim.Adam(encoder.parameters(), lr=hyper['lr'], weight_decay=hyper['l2'])
    attn_optim = optim.Adam(attn.parameters(), lr=hyper['lr'], weight_decay=hyper['l2'])
    ffnn_optim = optim.Adam(ffnn.parameters(), lr=hyper['lr'], weight_decay=hyper['l2'])
    if not args.nodecay:
        attn_scheduler = ReduceLROnPlateau(attn_optim, factor=hyper['dr'], patience=hyper['pat'], verbose=True, min_lr=0.00001)
        ffnn_scheduler = ReduceLROnPlateau(ffnn_optim, factor=hyper['dr'], patience=hyper['pat'], verbose=True, min_lr=0.00001)
    criterion = nn.MSELoss()
    model_target = [list(x[0][1].view(-1).data.cpu().numpy()) for x in test_vars]

    if args.only_load:
        print("stopped before training")
        raise
    # Pre-train with movie dialog
    if args.movie:
        print("Start training on movie dialog:" + str(len(train_vars_movie)))
        for _ in range(args.me):
            if not hyper['sem'] == "emhn":
                plot_losses, tot_loss, etc = trainIters(hyper['sem'], train_vars_movie, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=0, epoch=1)
            else:
                plot_losses, tot_loss, etc = trainIters_sem(hyper['sem'], train_vars_movie, encoder, encoder_sem, attn, ffnn, encoder_optim, encoder_sem_optim,attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=0, epoch=1)
        print("Evalualte movie only trained model")
	
        if hyper['sem'] == "emhn":
            model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, hyper['sem'], encoder, attn, ffnn, criterion, input_dial, encoder_sem=encoder_sem, features=test_features)
        else:
            model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, hyper['sem'], encoder, attn, ffnn, criterion, input_dial, features=test_features)

        eval_input = [model_output[x] + model_target[x] for x in range(len(model_target))]
        acc, fval = print_eval_short(eval_input, args.th)

    # Train
    num_epoch = args.ep
    epoch_losses = []
    epoch_attns = []
    epoch_accs = []
    for ep in range(args.start_ep, args.start_ep+num_epoch):
        print("start training on dialog data:"+str(len(train_vars)))
        if not hyper['sem'] =='emhn':
            plot_losses, tot_loss, etc = trainIters(hyper['sem'], train_vars, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=ep, epoch=num_epoch, features=train_features)
        else:
            plot_losses, tot_loss, etc = trainIters_sem(hyper['sem'], train_vars, encoder, encoder_sem, attn, ffnn, encoder_optim, encoder_sem_optim, attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=ep, epoch=num_epoch, features=train_features)

        savePlot(plot_losses, ep)
        print("{}th epoch - Train loss: {:.4f}".format(ep + 1, tot_loss))

        # Evaluate
        if hyper['sem'] == "emhn":
            model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, hyper['sem'], encoder, attn, ffnn, criterion, input_dial, encoder_sem=encoder_sem, features=test_features)
        else:
            model_loss, model_output, model_attn, model_context = evaluate_model(test_vars, hyper['sem'], encoder, attn, ffnn, criterion, input_dial, features=test_features)

        print("{}th epoch - Test loss: {:.4f}".format(ep + 1, model_loss))
        eval_input = [model_output[x] + model_target[x] for x in range(len(model_target))]
        acc, fval = print_eval_short(eval_input, args.th)

        epoch_accs.append(acc)
        epoch_losses.append(model_loss)
        epoch_attns.append(model_attn[0:5])

        if acc ==max(epoch_accs):
            best_acc = acc
            best_fval = fval
            best_loss = model_loss
            best_acc_output = eval_input
            if hyper['sem'] == "emhn":
                best_states = {
                    'epoch': ep,
                    'sem': hyper['sem'],
                    'encoder': encoder.state_dict(),
                    'encoder_optim': encoder_optim.state_dict(),
                    'encoder_sem': encoder_sem.state_dict(),
                    'encoder_sem_optim': encoder_sem_optim.state_dict(),
                    'attn': attn.state_dict(),
                    'attn_optim': attn_optim.state_dict(),
                    'ffnn': ffnn.state_dict(),
                    'ffnn_optim': ffnn_optim.state_dict(),
                    'args': args
                }
            else:
                best_states = {
                    'epoch': ep,
                    'sem': hyper['sem'],
                    'encoder': encoder.state_dict(),
                    'encoder_optim': encoder_optim.state_dict(),
                    'attn': attn.state_dict(),
                    'attn_optim': attn_optim.state_dict(),
                    'ffnn': ffnn.state_dict(),
                    'ffnn_optimizer': ffnn_optim.state_dict(),
                    'args': args
                }
        if not args.nodecay:
            attn_scheduler.step(model_loss)
            ffnn_scheduler.step(model_loss)
        if model_loss == min(epoch_losses):
            best_loss_output = eval_input

    print("Best accuracy model's result")
    print_eval(best_acc_output, args.th)
    if args.check:
        # last epoch
        if hyper['sem'] == "emhn":
            last_states = {
                'epoch': ep,
                'sem': hyper['sem'],
                'encoder': encoder.state_dict(),
                'encoder_optim': encoder_optim.state_dict(),
                'encoder_sem': encoder_sem.state_dict(),
                'encoder_sem_optim': encoder_sem_optim.state_dict(),
                'attn': attn.state_dict(),
                'attn_optim': attn_optim.state_dict(),
                'ffnn': ffnn.state_dict(),
                'ffnn_optim': ffnn_optim.state_dict(),
            }
        else:
            last_states = {
                'epoch': ep,
                'sem': hyper['sem'],
                'encoder': encoder.state_dict(),
                'encoder_optim': encoder_optim.state_dict(),
                'attn': attn.state_dict(),
                'attn_optim': attn_optim.state_dict(),
                'ffnn': ffnn.state_dict(),
                'ffnn_optim': ffnn_optim.state_dict(),
            }

        # save checkpoint
        timestr = time.strftime("%d%H%M")
        save_checkpoint(best_states, filename='./checkpoints/'+timestr+'_checkpoint'+str(args.start_ep+num_epoch)+'.tar')
        save_checkpoint(last_states, filename='./checkpoints/'+timestr+'_best_checkpoint'+str(args.start_ep+num_epoch)+'.tar')
    if len(set([str(x) for x in model_output])) == 1:
        print("Every output is the same. Retry.")
    else:
        timestr = time.strftime("%m%d_%H%M%S")
        with open('./loss_figures/' + timestr + '_' + str(hyper['lr']) + '_losses.txt', 'w') as file:
            file.write(str(epoch_losses))
        with open('./loss_figures/' + timestr + '_' + str(hyper['lr']) + '_attns.txt', 'w') as file:
            file.write(str(epoch_attns))

    print(epoch_losses)
    print(epoch_attns[-2:])

    return (best_acc, best_fval, best_loss, epoch_losses)
    # print("Last epoch's result")
    # print_eval(eval_input, args.th)
    # print("Best loss model's result")
    # print_eval(best_loss_output, args.th)
    # print("Best accuracy model's result")
    # print_eval(best_acc_output, args.th)

    # timestr = time.strftime("%m%d_%H%M%S")
    # with open('./loss_figures/' + timestr + '_' + str(args.lr) + '_losses.txt', 'w') as file:
    #     file.write(str(epoch_losses))
    # with open('./loss_figures/' + timestr + '_' + str(args.lr) + '_attns.txt', 'w') as file:
    #     file.write(str(epoch_attns))

# grid components

# h = [25, 50, 100, 200]
h_size = [250]
f_size = [64]
lr = [0.001, 0.0005]
sem_method = ['emh']
dr = [0.1, 0.5]
pat = [1, 2]
dp = [0.1]
l2 = [0.0005]
gl = [100]
attributes = {"h_size":h_size, "f_size":f_size, "lr":lr, "sem_method":sem_method, "dr":dr, "pat":pat,"dp":dp,"l2":l2}


if __name__ == "__main__":
    # build hyper dict
    hyper_dict_list = []

    for _h in h_size:
        for _gl in gl:
            for _pat in pat:
                for _sem in sem_method:
                    for _dr in dr:
                        for _f in f_size:
                            for _dp in dp:
                                for _l2 in l2:
                                    for _lr in lr:
                                        hyper_dict = {"h":_h, "ff":_f, "lr":_lr, "sem":_sem, "dr":_dr, "pat":_pat,"dp":_dp, "l2":_l2, "glove_dim":_gl}
                                        hyper_dict_list.append(hyper_dict)
    
    model_hyper = []
    model_res = []
    model_best = []
    for _dict in hyper_dict_list:
        print(_dict)
        res = one_model(_dict, args, train_vars, test_vars, dial_list, input_dial, output_class, pairs, train_features, test_features)
        model_hyper.append(_dict)
        model_res.append(res)
        model_best.append(res[0])
    model_zip = list(zip(model_hyper, model_res))
    sorted_model = [x for _,x in sorted(zip(model_best,model_zip), key=lambda pair: pair[0], reverse=True)]
    timestr = time.strftime("%m%d_%H%M%S")
    print(sorted_model[:5])
    with open('./loss_figures/' + timestr + '_grid_' + '.txt', 'w') as file:
        file.write(str(sorted_model))
