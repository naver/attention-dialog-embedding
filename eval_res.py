import ast
import math


def annotator_to_prob(inp_list):
    return [x / sum(inp_list) for x in inp_list]


def majority_label(prob, threshold):
    prob_O = prob[0]
    prob_T = prob[1]
    prob_X = prob[2]

    if prob_O >= prob_T and prob_O >= prob_X and prob_O >= threshold:
        return "O"
    elif prob_T >= prob_O and prob_T >= prob_X and prob_T >= threshold:
        return "T"
    elif prob_X >= prob_T and prob_X >= prob_O and prob_X >= threshold:
        return "X"
    else:
        return "O"


def majority_label_lenient(prob, threshold):
    prob_O = prob[0]
    prob_T = prob[1]
    prob_X = prob[2]

    if prob_O >= prob_T + prob_X and prob_O >= threshold:
        return "O_l"
    elif prob_T + prob_X >= prob_O and prob_T + prob_X >= threshold:
        return "X_l"
    else:
        return "O_l"


def kld(p, q):
    k = 0.0
    for i in range(len(p)):
        if p[i] > 0 and q[i] > 0:
            k += p[i] * (math.log(p[i] / q[i], 2))

    return k


def jsd(p, q):
    m = []
    for i in range(len(p)):
        m.append((p[i] + q[i]) / 2.0)

    return (kld(p, m) + kld(q, m)) / 2.0


def mse(p, q):
    total = 0.0

    for i in range(len(p)):
        total += pow(p[i] - q[i], 2)

    return total / len(p)


def grade_prediction(p, t, threshold):
    ans_label = majority_label(t, threshold)
    ans_label_l = majority_label_lenient(t, threshold)

    pred_label = majority_label(p, threshold)
    pred_label_l = majority_label_lenient(p, threshold)

    answer = ""
    res = ""
    res_l = ""

    if pred_label == ans_label:
        answer = "correct"
    else:
        answer = "incorrect"

    if pred_label == "O":
        if ans_label == "O":
            res = "predO_ansO"
        elif ans_label == "T":
            res = "predO_ansT"
        elif ans_label == "X":
            res = "predO_ansX"
    elif pred_label == "T":
        if ans_label == "O":
            res = "predT_ansO"
        elif ans_label == "T":
            res = "predT_ansT"
        elif ans_label == "X":
            res = "predT_ansX"
    elif pred_label == "X":
        if ans_label == "O":
            res = "predX_ansO"
        elif ans_label == "T":
            res = "predX_ansT"
        elif ans_label == "X":
            res = "predX_ansX"

    # lenient
    if pred_label == "O":
        if ans_label_l == "O_l":
            res_l = "predO_ansO_l"
        elif ans_label_l == "X_l":
            res_l = "predO_ansX_l"
    elif pred_label == "T":
        if ans_label_l == "O_l":
            res_l = "predT_ansO_l"
        elif ans_label_l == "X_l":
            res_l = "predT_ansX_l"
    elif pred_label == "X":
        if ans_label_l == "O_l":
            res_l = "predX_ansO_l"
        elif ans_label_l == "X_l":
            res_l = "predX_ansX_l"

    return answer, res, res_l


if __name__ == "__main__":
    with open('pred_tgt_prob.txt') as f:
        lines = f.read().splitlines()

    lines = [ast.literal_eval(x) for x in lines]


def print_eval(lines, th):
    pred = [x[0:3] for x in lines]
    tgt = [x[3:6] for x in lines]

    correct = 0
    incorrect = 0

    predO_ansO = 0
    predO_ansT = 0
    predO_ansX = 0
    predT_ansO = 0
    predT_ansT = 0
    predT_ansX = 0
    predX_ansO = 0
    predX_ansT = 0
    predX_ansX = 0

    predO_ansO_l = 0
    predO_ansX_l = 0
    predT_ansO_l = 0
    predT_ansX_l = 0
    predX_ansO_l = 0
    predX_ansX_l = 0

    avg_jsd = 0
    avg_mse = 0

    jsd_O_T_X_sum = 0.0
    jsd_O_TX_sum = 0.0
    jsd_OT_X_sum = 0.0
    mse_O_T_X_sum = 0.0
    mse_O_TX_sum = 0.0
    mse_OT_X_sum = 0.0

    total = len(pred)
    for i in range(total):
        pred_prob = pred[i]
        tgt_prob = tgt[i]
        # pred_prob = annotator_to_prob(pred[i])
        # tgt_prob = annotator_to_prob(tgt[i])
        avg_jsd += jsd(pred_prob, tgt_prob)
        avg_mse += mse(pred_prob, tgt_prob)
        _ans, _res, _res_l = grade_prediction(pred_prob, tgt_prob, th)

        if _ans == "correct":
            correct += 1
        elif _ans == "incorrect":
            incorrect += 1

        if _res == "predO_ansO":
            predO_ansO += 1
        elif _res == "predO_ansT":
            predO_ansT += 1
        elif _res == "predO_ansX":
            predO_ansX += 1
        elif _res == "predT_ansO":
            predT_ansO += 1
        elif _res == "predT_ansT":
            predT_ansT += 1
        elif _res == "predT_ansX":
            predT_ansX += 1
        elif _res == "predX_ansO":
            predX_ansO += 1
        elif _res == "predX_ansT":
            predX_ansT += 1
        elif _res == "predX_ansX":
            predX_ansX += 1

        if _res_l == "predO_ansO_l":
            predO_ansO_l += 1
        elif _res_l == "predO_ansX_l":
            predO_ansX_l += 1
        elif _res_l == "predT_ansO_l":
            predT_ansO_l += 1
        elif _res_l == "predT_ansX_l":
            predT_ansX_l += 1
        elif _res_l == "predX_ansO_l":
            predX_ansO_l += 1
        elif _res_l == "predX_ansX_l":
            predX_ansX_l += 1

        jsd_O_T_X_sum += jsd(tgt_prob, pred_prob)
        jsd_O_TX_sum += jsd([tgt_prob[0], tgt_prob[1] + tgt_prob[2]], [pred_prob[0], pred_prob[1] + pred_prob[2]])
        jsd_OT_X_sum += jsd([tgt_prob[0] + tgt_prob[1], tgt_prob[2]], [pred_prob[0] + pred_prob[1], pred_prob[2]])

        mse_O_T_X_sum += mse(tgt_prob, pred_prob)
        mse_O_TX_sum += mse([tgt_prob[0], tgt_prob[1] + tgt_prob[2]], [pred_prob[0], pred_prob[1] + pred_prob[2]])
        mse_OT_X_sum += mse([tgt_prob[0] + tgt_prob[1], tgt_prob[2]], [pred_prob[0] + pred_prob[1], pred_prob[2]])

    try:
        avg_jsd = avg_jsd / total
        avg_mse = avg_mse / total
    except:
        "Error: total=0"

    print("avg. jsd is: " + str(avg_jsd))
    print("avg. mse is: " + str(avg_mse))

    print("######### Data Stats #########")
    print("O Label Num : \t\t" + str(predO_ansO + predT_ansO + predX_ansO))
    print("T Label Num : \t\t" + str(predO_ansT + predT_ansT + predX_ansT))
    print("X Label Num : \t\t" + str(predO_ansX + predT_ansX + predX_ansX))
    print("O Predict Num : \t\t" + str(predO_ansO + predO_ansT + predO_ansX))
    print("T Predict Num : \t\t" + str(predT_ansO + predT_ansT + predT_ansX))
    print("X Predict Num : \t\t" + str(predX_ansO + predX_ansT + predX_ansX))
    print("")

    print("######### Results #########")
    print("Accuracy : \t\t%.4f" % ((correct * 1.0) / (correct + incorrect)) +
          " (" + str(correct) + "/" + str(correct + incorrect) + ")\n")
    precision_s = 0.0
    recall_s = 0.0
    fmeasure_s = 0.0

    if predX_ansX > 0:
        if (predX_ansO + predX_ansT + predX_ansX) > 0:
            precision_s = predX_ansX * 1.0 / (predX_ansO + predX_ansT + predX_ansX)
        if (predO_ansX + predT_ansX + predX_ansX) > 0:
            recall_s = predX_ansX * 1.0 / (predO_ansX + predT_ansX + predX_ansX)

    if precision_s > 0 and recall_s > 0:
        fmeasure_s = (2 * precision_s * recall_s) / (precision_s + recall_s)

    print("Precision (X) : \t%.4f" % (precision_s) + " (" + str(predX_ansX) + "/" + str(predX_ansO + predX_ansT + predX_ansX) + ")")
    print("Recall    (X) : \t%.4f" % (recall_s) + " (" + str(predX_ansX) + "/" + str(predO_ansX + predT_ansX + predX_ansX) + ")")
    print("F-measure (X) : \t%.4f" % (fmeasure_s) + "\n")

    precision_l = 0.0
    recall_l = 0.0
    fmeasure_l = 0.0

    if (predT_ansX_l + predX_ansX_l) > 0 and (predX_ansO_l + predX_ansX_l + predT_ansO_l + predT_ansX_l) > 0:
        precision_l = (predT_ansX_l + predX_ansX_l) * 1.0 / (predX_ansO_l + predX_ansX_l + predT_ansO_l + predT_ansX_l)
    if (predT_ansX_l + predX_ansX_l) > 0 and (predO_ansX_l + predT_ansX_l + predX_ansX_l) > 0:
        recall_l = (predT_ansX_l + predX_ansX_l) * 1.0 / (predO_ansX_l + predT_ansX_l + predX_ansX_l)
    if precision_l > 0 and recall_l > 0:
        fmeasure_l = (2 * precision_l * recall_l) / (precision_l + recall_l)

    print("Precision (T+X) : \t%.4f" % (precision_l) + " (" + str(predT_ansX_l + predX_ansX_l) +
          "/" + str(predX_ansO_l + predX_ansX_l + predT_ansO_l + predT_ansX_l) + ")")
    print("Recall    (T+X) : \t%.4f" % (recall_l) + " (" + str(predT_ansX_l + predX_ansX_l) +
          "/" + str(predO_ansX_l + predT_ansX_l + predX_ansX_l) + ")")
    print("F-measure (T+X) : \t%.4f" % (fmeasure_l) + "\n")

    print("JS divergence (O,T,X) : \t%.4f" % (jsd_O_T_X_sum / (correct + incorrect)))
    print("JS divergence (O,T+X) : \t%.4f" % (jsd_O_TX_sum / (correct + incorrect)))
    print("JS divergence (O+T,X) : \t%.4f" % (jsd_OT_X_sum / (correct + incorrect)) + "\n")

    print("Mean squared error (O,T,X) : \t%.4f" % (mse_O_T_X_sum / (correct + incorrect)))
    print("Mean squared error (O,T+X) : \t%.4f" % (mse_O_TX_sum / (correct + incorrect)))
    print("Mean squared error (O+T,X) : \t%.4f" % (mse_OT_X_sum / (correct + incorrect)))
    print(predO_ansO, predO_ansT, predO_ansX, predT_ansO, predT_ansT, predT_ansX, predX_ansO, predX_ansT, predX_ansX)
    print("###########################")



    return None

def print_eval_short(lines, th):
    pred = [x[0:3] for x in lines]
    tgt = [x[3:6] for x in lines]

    correct = 0
    incorrect = 0

    predO_ansO = 0
    predO_ansT = 0
    predO_ansX = 0
    predT_ansO = 0
    predT_ansT = 0
    predT_ansX = 0
    predX_ansO = 0
    predX_ansT = 0
    predX_ansX = 0

    predO_ansO_l = 0
    predO_ansX_l = 0
    predT_ansO_l = 0
    predT_ansX_l = 0
    predX_ansO_l = 0
    predX_ansX_l = 0

    avg_jsd = 0
    avg_mse = 0

    jsd_O_T_X_sum = 0.0
    jsd_O_TX_sum = 0.0
    jsd_OT_X_sum = 0.0
    mse_O_T_X_sum = 0.0
    mse_O_TX_sum = 0.0
    mse_OT_X_sum = 0.0

    total = len(pred)
    for i in range(total):
        pred_prob = pred[i]
        tgt_prob = tgt[i]
        # pred_prob = annotator_to_prob(pred[i])
        # tgt_prob = annotator_to_prob(tgt[i])
        avg_jsd += jsd(pred_prob, tgt_prob)
        avg_mse += mse(pred_prob, tgt_prob)
        _ans, _res, _res_l = grade_prediction(pred_prob, tgt_prob, th)

        if _ans == "correct":
            correct += 1
        elif _ans == "incorrect":
            incorrect += 1

        if _res == "predO_ansO":
            predO_ansO += 1
        elif _res == "predO_ansT":
            predO_ansT += 1
        elif _res == "predO_ansX":
            predO_ansX += 1
        elif _res == "predT_ansO":
            predT_ansO += 1
        elif _res == "predT_ansT":
            predT_ansT += 1
        elif _res == "predT_ansX":
            predT_ansX += 1
        elif _res == "predX_ansO":
            predX_ansO += 1
        elif _res == "predX_ansT":
            predX_ansT += 1
        elif _res == "predX_ansX":
            predX_ansX += 1

        if _res_l == "predO_ansO_l":
            predO_ansO_l += 1
        elif _res_l == "predO_ansX_l":
            predO_ansX_l += 1
        elif _res_l == "predT_ansO_l":
            predT_ansO_l += 1
        elif _res_l == "predT_ansX_l":
            predT_ansX_l += 1
        elif _res_l == "predX_ansO_l":
            predX_ansO_l += 1
        elif _res_l == "predX_ansX_l":
            predX_ansX_l += 1

        jsd_O_T_X_sum += jsd(tgt_prob, pred_prob)
        jsd_O_TX_sum += jsd([tgt_prob[0], tgt_prob[1] + tgt_prob[2]], [pred_prob[0], pred_prob[1] + pred_prob[2]])
        jsd_OT_X_sum += jsd([tgt_prob[0] + tgt_prob[1], tgt_prob[2]], [pred_prob[0] + pred_prob[1], pred_prob[2]])

        mse_O_T_X_sum += mse(tgt_prob, pred_prob)
        mse_O_TX_sum += mse([tgt_prob[0], tgt_prob[1] + tgt_prob[2]], [pred_prob[0], pred_prob[1] + pred_prob[2]])
        mse_OT_X_sum += mse([tgt_prob[0] + tgt_prob[1], tgt_prob[2]], [pred_prob[0] + pred_prob[1], pred_prob[2]])

    try:
        avg_jsd = avg_jsd / total
        avg_mse = avg_mse / total
    except:
        "Error: total=0"

    print("######### Results #########")
    print("Label Num (O,T,X): {}, {}, {}\t".format(predO_ansO + predT_ansO + predX_ansO, predO_ansT + predT_ansT + predX_ansT, predO_ansX + predT_ansX + predX_ansX))
    print("Predict Num (O,T,X): {}, {}, {}\t".format(predO_ansO + predO_ansT + predO_ansX, predT_ansO + predT_ansT + predT_ansX, predX_ansO + predX_ansT + predX_ansX))
    accuracy = (correct * 1.0) / (correct + incorrect)
    print("Accuracy : \t\t%.4f" % (accuracy) +
          " (" + str(correct) + "/" + str(correct + incorrect) + ")\n")
    precision_s = 0.0
    recall_s = 0.0
    fmeasure_s = 0.0

    if predX_ansX > 0:
        if (predX_ansO + predX_ansT + predX_ansX) > 0:
            precision_s = predX_ansX * 1.0 / (predX_ansO + predX_ansT + predX_ansX)
        if (predO_ansX + predT_ansX + predX_ansX) > 0:
            recall_s = predX_ansX * 1.0 / (predO_ansX + predT_ansX + predX_ansX)

    if precision_s > 0 and recall_s > 0:
        fmeasure_s = (2 * precision_s * recall_s) / (precision_s + recall_s)

    print("Precision (X) : \t%.4f" % (precision_s) + " (" + str(predX_ansX) + "/" + str(predX_ansO + predX_ansT + predX_ansX) + ")")
    print("Recall    (X) : \t%.4f" % (recall_s) + " (" + str(predX_ansX) + "/" + str(predO_ansX + predT_ansX + predX_ansX) + ")")
    print("F-measure (X) : \t%.4f" % (fmeasure_s) + "\n")
    print("JS divergence (O,T,X) : \t%.4f" % (jsd_O_T_X_sum / (correct + incorrect)))
    print("Mean squared error (O,T,X) : \t%.4f" % (mse_O_T_X_sum / (correct + incorrect)))
    print(predO_ansO, predO_ansT, predO_ansX, predT_ansO, predT_ansT, predT_ansX, predX_ansO, predX_ansT, predX_ansX)
    print("###########################")

    return accuracy, fmeasure_s
