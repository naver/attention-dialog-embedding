from torch.autograd import Variable
from utils import *
from settings import *
from trainer import model_output

def evaluate(args, encoder, attn_sum, ffnn, criterion, input_variable_list, target_var, input_dial, max_length=MAX_LENGTH, max_context=MAX_CONTEXT, encoder_emhn=None, feature=None):
    output, attn_weights, context_vector = model_output(args, input_variable_list, encoder, attn_sum, ffnn, max_context, max_length, encoder_emhn, feature)
    loss = criterion(output, target_var)
    return output, loss.data[0], attn_weights, context_vector


def predict(args, encoder, attn_sum, ffnn, input_variable_list, input_dial, max_length=MAX_LENGTH, max_context=MAX_CONTEXT, encoder_emhn=None, feature=None):
    output, attn_weights, context_vector = model_output(args, input_variable_list, encoder, attn_sum, ffnn, max_context, max_length, encoder_emhn, feature)
    return output 


def evaluateRandomly(args, encoder, attn_sum, ffnn, criterion, input_dial, pairs, dial_list, n=1, encoder_emhn=None):
    for i in range(n):
        _text, _pair, _input_variable_list = random_pair_var_list(pairs, dial_list, input_dial)
        print('>', _text)
        print('=', _pair[1].view(1, -1))
        output, loss, attn_weights, context_vec = evaluate(args, encoder, attn_sum, ffnn, criterion,_input_variable_list, _pair[1], input_dial, encoder_emhn=encoder_emhn)
        print('<', output)


def evaluate_model(args, data, encoder, attn_sum, ffnn, criterion, input_dial, encoder_emhn=None, features=None):
    tot_loss = 0
    tot_output = []
    tot_attn_weights = []
    tot_context_vec = []
    for idx, pair in enumerate(data):
        input_variable_list = pair[1]
        target_variable = pair[0][1]
        if use_cuda:
            input_variable_list = [var.cuda() for var in input_variable_list]
            target_variable = target_variable.cuda()
        if features is None:
            output, loss, attn_weights, context_vec = evaluate(args, encoder, attn_sum, ffnn, criterion, input_variable_list, target_variable, input_dial, encoder_emhn=encoder_emhn)
        else:
            output, loss, attn_weights, context_vec = evaluate(args, encoder, attn_sum, ffnn, criterion, input_variable_list, target_variable, input_dial, encoder_emhn=encoder_emhn, feature=features[idx])

        tot_loss += loss
        #tot_output.append(list(output.data.cpu().numpy()[0]))
        #tot_attn_weights.append(list(attn_weights.data.cpu().numpy()[0]))
        tot_output.append(list(output.data.cpu().numpy()))
        try:
            tot_attn_weights.append(list(attn_weights.data.cpu().numpy()[0]))
        except:
            tot_attn_weights.append(None)
        tot_context_vec.append(context_vec)
    return tot_loss / len(data), tot_output, tot_attn_weights, tot_context_vec

