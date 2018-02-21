import chainer
from chainer import cuda

xp = cuda.cupy

def _correct_counter(pred_selection, l):
    correct = 0
    reach = 0
    for i in range(len(l)-1):
        if l[i+1] == pred_selection[i+1] == 1:
            if reach == 1:
                correct += 1
            else:
                reach = 1
        if l[i+1] != pred_selection[i+1] and reach == 1:
            reach = 0
    return correct

def _divide_correct_counter(pred_selection, l):
    correct = 0
    for i in range(len(pred_selection)):
        if pred_selection[i] == l[i]:
            correct += 1
    return correct - 2
        
def eval(data, model):
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        pred_word_count = 0
        gold_word_count = 0
        correct_word_count = 0
        divide_correct = 0
        all_sentence_len = 0
        for i in range(len(data[0])):
            x = [xp.asarray(data[0][i], dtype=xp.int32)]
            l = data[1][i]
            loss, pred_labels = model(x, [l])
            pred_selection = pred_labels[0]
            pred_word_count += sum(pred_selection) - 2
            gold_word_count += sum(l) -2
            correct_word_count += _correct_counter(pred_selection, l)
            divide_correct += _divide_correct_counter(pred_selection, l)
            all_sentence_len += len(pred_selection) - 2
    return pred_word_count, gold_word_count, correct_word_count, divide_correct/all_sentence_len

def recorder(data, epoch, path, model):
    pred_word_count, gold_word_count, correct_word_count, accuracy = eval(data, model)
    print("accuracy", accuracy)
    print("pred_word_count", pred_word_count, "gold_word_count", gold_word_count, "correct_word_count", correct_word_count)
    if pred_word_count <= 0:
        print("pred_word 0")
        pred_word_count = 1        
    precision = correct_word_count/pred_word_count
    if correct_word_count == 0:
        print("correct 0")
        correct_word_count = 1
    recall = correct_word_count/gold_word_count
    print("precision", precision, "recall", recall)
    Fscore = 2*recall*precision/(recall+precision)
    print("Fscore", Fscore)
    
    w = open(path, "a")
    w.write("epoch {}\n".format(str(epoch)))
    w.write("accuracy {}\n".format(str(accuracy)))
    w.write("pred_word_count {} gold_word_count {} correct_word_count {}\n".format(str(pred_word_count), str(gold_word_count), str(correct_word_count)))
    w.write("precision {} recall {}\n".format(str(precision), str(recall)))
    w.write("Fscore {}\n".format(str(Fscore)))
    w.write("-------------------------------------------------------\n")
    w.close()
