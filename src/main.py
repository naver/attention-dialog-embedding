import torch
import random
import os
import pickle
from parser import args
from utils import *
from models import *
from trainer import *
from evaluater import *
from torch import optim
from settings import *
from eval_res import print_eval, print_eval_short
from torch.optim.lr_scheduler import *

# load data
# if args.data == "all":
dial_list, input_dial, output_class, pairs = prepareData('./data/dial_list_all_unk.txt', './data/tgt_list_all_onehot.txt')
#elif args.data == "it":
#    dial_list, input_dial, output_class, pairs = prepareData('./data/dial_list_unk.txt', './data/tgt_list.txt')
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

if not args.data == "all":
    features=None
    if not args.no_feature:
        with open('./data/features', 'rb') as file:
            features = pickle.load(file)
    if args.ran:
        train_tuple, test_tuple, i_tuple, t_tuple, c_tuple, y_tuple = load_separate_data_vars(dial_list, pairs, features)
    else:
        train_tuple, test_tuple, i_tuple, t_tuple, c_tuple, y_tuple = import_separate_data_vars(dial_list, pairs, features)
    train_vars, train_features = train_tuple
    test_vars, test_features = test_tuple
    i_train_vars, i_test_vars, i_train_features, i_test_features = i_tuple
    t_train_vars, t_test_vars, t_train_features, t_test_features = t_tuple
    c_train_vars, c_test_vars, c_train_features, c_test_features = c_tuple
    y_train_vars, y_test_vars, y_train_features, y_test_features = y_tuple

else: 
    if args.no_feature:
        if args.ran:
		# split data to train, test data if option selcted as random
            train_vars, test_vars = split_data(pairs, args.split_p, dial_list, input_dial)
            train_features, test_features = None, None
        else:
            train_vars = pickle.load(open('./data/train_vars_all_1', 'rb'))
            test_vars = pickle.load(open('./data/test_vars_all_1', 'rb'))
            train_features, test_features = None, None
    else:
        with open('./data/features_16', 'rb') as file:
            features = pickle.load(file)
        if args.ran:
		# split data to train, test data if option selcted as random
            train_vars, test_vars, train_features, test_features= split_data_features(pairs, args.split_p, dial_list, input_dial, features)
        else:
            train_vars = pickle.load(open('./data/train_vars_all_1', 'rb'))
            test_vars = pickle.load(open('./data/test_vars_all_1', 'rb'))
            train_features = pickle.load(open('./data/train_features_all_1', 'rb'))
            test_features = pickle.load(open('./data/test_features_all_1', 'rb'))
        if args.no_split:
            train_vars = train_vars + test_vars
            train_features = train_features + test_features
        train_vars = list(zip(train_vars, train_features))
        test_vars = list(zip(test_vars, test_features))

if args.movie or args.mix_movie:
    if os.path.isfile('./data/movie_train_vars'):
        with open('./data/movie_train_vars','rb') as file:
            train_vars_movie = pickle.load(file)
        print("loaded pre-saved movie_train_vars - len: "+str(len(train_vars_movie)))
    else:
        train_vars_movie = split_data_movie(pairs_movie, args.split_p, dial_list_movie, input_dial_movie)
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

if args.test:
    train_vars = random.sample(train_vars, 100)
    test_vars = random.sample(test_vars, 50)
if args.toy:
    train_vars = random.sample(train_vars, 1000)
    test_vars = random.sample(test_vars, 100)

random.shuffle(train_vars)
random.shuffle(test_vars)
if not args.data=="all":
    random.shuffle(i_train_vars)
    random.shuffle(i_test_vars)
    random.shuffle(t_train_vars)
    random.shuffle(t_test_vars)
    random.shuffle(c_train_vars)
    random.shuffle(c_test_vars)
    random.shuffle(y_train_vars)
    random.shuffle(y_test_vars)
if not args.no_feature:
    train_zip = list(zip(*train_vars))
    train_vars = list(train_zip[0])
    train_features = list(train_zip[1])
    test_zip = list(zip(*test_vars))
    test_vars = list(test_zip[0])
    test_features = list(test_zip[1])
    if not args.data == "all":
        i_train_zip = list(zip(*i_train_vars))
        i_train_vars = list(i_train_zip[0])
        i_train_features = list(i_train_zip[1])
        i_test_zip = list(zip(*i_test_vars))
        i_test_vars = list(i_test_zip[0])
        i_test_features = list(i_test_zip[1])
        t_train_zip = list(zip(*t_train_vars))
        t_train_vars = list(t_train_zip[0])
        t_train_features = list(t_train_zip[1])
        t_test_zip = list(zip(*t_test_vars))
        t_test_vars = list(t_test_zip[0])
        t_test_features = list(t_test_zip[1])
        c_train_zip = list(zip(*c_train_vars))
        c_train_vars = list(c_train_zip[0])
        c_train_features = list(c_train_zip[1])
        c_test_zip = list(zip(*c_test_vars))
        c_test_vars = list(c_test_zip[0])
        c_test_features = list(c_test_zip[1])
        y_train_zip = list(zip(*y_train_vars))
        y_train_vars = list(y_train_zip[0])
        y_train_features = list(y_train_zip[1])
        y_test_zip = list(zip(*y_test_vars))
        y_test_vars = list(y_test_zip[0])
        y_test_features = list(y_test_zip[1])
            
# Technical DEBTH
if args.data == "i":
        train_vars, test_vars = i_train_vars, i_test_vars
        if not args.no_feature:
            train_features, test_features = i_train_features, i_test_features
elif args.data == "t":
        train_vars, test_vars = t_train_vars, t_test_vars
        if not args.no_feature:
            train_features, test_features = t_train_features, t_test_features
elif args.data == "c":
        train_vars, test_vars = c_train_vars, c_test_vars
        if not args.no_feature:
            train_features, test_features = c_train_features, c_test_features
elif args.data == "y":
        train_vars, test_vars = y_train_vars, y_test_vars
        if not args.no_feature:
            train_features, test_features = y_train_features, y_test_features
 

# load args
if args.resume:
    if os.path.isfile('./checkpoints/'+args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load('./checkpoints/'+args.resume)
        args = checkpoint['args']
# define models, optimizers, loss criterion
encoder = Encoder_glove(input_dial.glove_voc, input_dial.glove_vec, args.hs,n_layers=ENC_LAYER)
encoder_emhn = None
if args.sem == 'emh':
	attn = AttnSum(2*(input_dial.glove_vec.size()[1] +args.hs), args.sem, dropout_p=args.dp)
	ffnn = FFNN(attn.hs, args.ff, output_class.n_class)
elif args.sem == 'h': 
	attn = AttnSum(args.hs*2, args.sem, dropout_p=args.dp)
	ffnn = FFNN(input_dial.glove_vec.size()[1]+args.hs, args.ff, output_class.n_class)
elif args.sem =='em':
	attn = AttnSum(input_dial.glove_vec.size()[1] +args.hs*1, args.sem, dropout_p=args.dp)
	ffnn = FFNN(input_dial.glove_vec.size()[1] *2, args.ff, output_class.n_class)
elif args.sem == 'emhn':
	encoder_emhn = Encoder_emhn(input_dial.glove_vec.size()[1], input_dial.glove_vec.size()[1], n_layers=1)
	encoder_emhn_optim = optim.Adam(encoder_emhn.parameters(), lr=args.lr, weight_decay=args.l2)
	encoder_emhn = encoder_emhn.cuda() if use_cuda else encoder_emhn
	attn = AttnSum(2*(input_dial.glove_vec.size()[1] *2 + args.hs), args.sem, dropout_p=args.dp)
	ffnn = FFNN(attn.hs, args.ff, output_class.n_class)
if not args.no_feature:
    ffnn = FFNN(ffnn.input_size+FEATURE_SIZE, ffnn.hidden_size, output_class.n_class)
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
if args.resume:
    if os.path.isfile('./checkpoints/'+args.resume):
        args.start_ep = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['encoder'])
        encoder_optim.load_state_dict(checkpoint['encoder_optim'])
        if args.sem =="emhn":
            encoder_emhn.load_state_dict(checkpoint['encoder_emhn'])
            encoder_emhn_optim.load_state_dict(checkpoint['encoder_emhn_optim'])
        attn.load_state_dict(checkpoint['attn'])
        attn_optim.load_state_dict(checkpoint['attn_optim'])
        ffnn.load_state_dict(checkpoint['ffnn'])
        ffnn_optim.load_state_dict(checkpoint['ffnn_optim'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
if args.criterion == "mse":
    criterion = nn.MSELoss()
elif args.criterion == "multisoft":
    criterion = nn.MultiLabelSoftMarginLoss()
model_target = [list(x[0][1].view(-1).data.cpu().numpy()) for x in test_vars]

if args.only_load:
    print("stopped before training")
    raise
# Pre-train with movie dialog
if args.movie:
    print("Start training on movie dialog:" + str(len(train_vars_movie)))
    for _ in range(args.me):
        plot_losses, tot_loss, etc = trainIters(args, train_vars_movie, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=0, epoch=1, encoder_emhn=encoder_emhn)
    print("Evalualte movie only trained model")

    model_loss, model_output, model_attn, model_context = evaluate_model(args, test_vars, encoder, attn, ffnn, criterion, input_dial, encoder_emhn=encoder_emhn, features=test_features)
		
    eval_input = [model_output[x] + model_target[x] for x in range(len(model_target))]
    acc, fval = print_eval_short(eval_input, args.th)

# Train
num_epoch = args.ep
epoch_losses = []
epoch_attns = []
epoch_accs = []
train_losses = []
total_plot_losses = []
for ep in range(args.start_ep, args.start_ep+num_epoch):
    print("start training on dialog data:"+str(len(train_vars)))
    plot_losses, tot_loss, etc = trainIters(args, train_vars, encoder, attn, ffnn, encoder_optim, attn_optim, ffnn_optim, criterion, print_every=args.pr, cur_epoch=ep, epoch=num_epoch, encoder_emhn=encoder_emhn, features=train_features)

    savePlot(plot_losses, ep)
    total_plot_losses.extend(plot_losses)
    train_losses.append(tot_loss)
    print("{}th epoch - Train loss: {:.4f}".format(ep + 1, tot_loss))

    # Evaluate
    if not args.no_split:
        model_loss, model_output, model_attn, model_context = evaluate_model(args, test_vars, encoder, attn, ffnn, criterion, input_dial, encoder_emhn=encoder_emhn, features=test_features)
	
        print("{}th epoch - Test loss: {:.4f}".format(ep + 1, model_loss))
        eval_input = [model_output[x] + model_target[x] for x in range(len(model_target))]
        acc, fval = print_eval_short(eval_input, args.th)
        epoch_accs.append(acc)
        epoch_losses.append(model_loss)
        epoch_attns.append(model_attn[0:5])
        if acc ==max(epoch_accs):
            best_acc_output = eval_input
            best_encoder, best_attn, best_ffnn = encoder, attn, ffnn
            if args.sem == "emhn":
                best_states = {
                    'epoch': ep,
                    'sem': args.sem,
                    'encoder': encoder.state_dict(),
                    'encoder_optim': encoder_optim.state_dict(),
                    'encoder_emhn': encoder_emhn.state_dict(),
                    'encoder_emhn_optim': encoder_emhn_optim.state_dict(),
                    'attn': attn.state_dict(),
                    'attn_optim': attn_optim.state_dict(),
                    'ffnn': ffnn.state_dict(),
                    'ffnn_optim': ffnn_optim.state_dict(),
                    'args': args
                }
            else:
                best_states = {
                    'epoch': ep,
                    'sem': args.sem,
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
    else:
        attn_scheduler.step(tot_loss)
        ffnn_scheduler.step(tot_loss)
        if tot_loss == min(train_losses):
            if args.sem == "emhn":
                best_states = {
                    'epoch': ep,
                    'sem': args.sem,
                    'encoder': encoder.state_dict(),
                    'encoder_optim': encoder_optim.state_dict(),
                    'encoder_emhn': encoder_emhn.state_dict(),
                    'encoder_emhn_optim': encoder_emhn_optim.state_dict(),
                    'attn': attn.state_dict(),
                    'attn_optim': attn_optim.state_dict(),
                    'ffnn': ffnn.state_dict(),
                    'ffnn_optim': ffnn_optim.state_dict(),
                    'args': args
                }
            else:
                best_states = {
                    'epoch': ep,
                    'sem': args.sem,
                    'encoder': encoder.state_dict(),
                    'encoder_optim': encoder_optim.state_dict(),
                    'attn': attn.state_dict(),
                    'attn_optim': attn_optim.state_dict(),
                    'ffnn': ffnn.state_dict(),
                    'ffnn_optimizer': ffnn_optim.state_dict(),
                    'args': args
                }
            
        model_loss, model_output, model_attn, model_context = None, None, None, None
        acc, fval = 0, 0
    
if args.submit:
    submit_vars = pickle.load(open('./data/train_vars_submit','rb'))
    submit_features = pickle.load(open('./data/features_submit_16','rb'))
    with open('data/tgt_submit_answer.txt','r') as file:
        answer = file.read()
    answer= ast.literal_eval(answer)
    answer = [[[z/sum(y) for z in y] for y in x] for x in answer]
    answer2 = []
    for x in answer:
        answer2.extend(x)
    res = []
    best_res = []
    for i in range(len(submit_vars)):
        res.append(predict(args, encoder, attn, ffnn, submit_vars[i][1], input_dial, encoder_emhn=encoder_emhn, feature=submit_features[i])) 
        best_res.append(predict(args, best_encoder, best_attn, best_ffnn, submit_vars[i][1], input_dial, encoder_emhn=encoder_emhn, feature=submit_features[i])) 
    res = [list(x.data) for x in res]
    timestr = time.strftime("%d%H%M")
    with open('./results/res_'+timestr+'_ep'+str(ep), 'wb') as file:
        pickle.dump(res, file)
    with open('./results/best_res_'+timestr+'_ep'+str(ep), 'wb') as file:
        pickle.dump(best_res, file)
    submit_eval_input = [res[x] + answer2[x] for x in range(len(answer))]
    print("\n****** Final Submission Result ******\n")
    print_eval(submit_eval_input, args.th)
savePlot(train_losses, '_train_epoch_total')    
savePlot(total_plot_losses, '_train_total')
if not args.no_split:
    savePlot(model_loss, '_test_total')
    print("Last epoch's result")
    print_eval(eval_input, args.th)
    print("Best loss model's result")
    print_eval(best_loss_output, args.th)
    print("Best accuracy model's result")
    print_eval(best_acc_output, args.th)
if args.check:
	# last epoch
	if args.sem == "emhn":
		last_states = {
			'epoch': ep,
			'sem': args.sem,
			'encoder': encoder.state_dict(),
			'encoder_optim': encoder_optim.state_dict(),
			'encoder_emhn': encoder_emhn.state_dict(),
			'encoder_emhn_optim': encoder_emhn_optim.state_dict(),
			'attn': attn.state_dict(),
			'attn_optim': attn_optim.state_dict(),
			'ffnn': ffnn.state_dict(),
			'ffnn_optim': ffnn_optim.state_dict(),
            'args': args
		}
	else:
		last_states = {
			'epoch': ep,
			'sem': args.sem,
			'encoder': encoder.state_dict(),
			'encoder_optim': encoder_optim.state_dict(),
			'attn': attn.state_dict(),
			'attn_optim': attn_optim.state_dict(),
			'ffnn': ffnn.state_dict(),
			'ffnn_optim': ffnn_optim.state_dict(),
            'args': args
		}

	# save checkpoint
	timestr = time.strftime("%d%H%M")
	save_checkpoint(last_states, filename='./checkpoints/'+timestr+'_checkpoint_hs'+str(args.hs)+"_ep_"+str(args.start_ep+num_epoch)+'.tar')
	save_checkpoint(best_states, filename='./checkpoints/'+timestr+'_best_checkpoint_hs'+str(args.hs)+"_ep_"+str(args.start_ep+num_epoch)+'.tar')
	timestr = time.strftime("%m%d_%H%M%S")
	with open('./loss_figures/' + timestr + '_' + str(args.lr) + '_losses.txt', 'w') as file:
		file.write(str(epoch_losses))
	with open('./loss_figures/' + timestr + '_' + str(args.lr) + '_attns.txt', 'w') as file:
		file.write(str(epoch_attns))
	print(epoch_losses)
	print(epoch_attns[-2:])

print(train_losses)

# Manual Check
# evaluateRandomly(encoder, attn, ffnn, criterion, input_dial, pairs, dial_list, n=2)
# predict(enc der2, attn_sum2, ffnn2, criterion2, ["hi","hi alex", "how are you", "i am good"], input_dial)
