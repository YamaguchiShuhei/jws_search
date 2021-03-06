# word head char = 1(sep), word other char = 0{app)
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import random
import sys
import pickle
import pdb # pdb.set_trace()
import time # time.time()

import data
import path
#import model


params = {"FREQ_times":3, "SENT_len":300, "embedding_size":100, "batch_size":10, "CHAR_size":3000,"hidden_size":150, "action_size":50, "prescore_size":300, "wordvec_size":200, "LSTM_units":1, "dropout":0.2, "margin_rate":0.2, "beam_size":8}

train_path, test_path, wiki_path, dict_path = path.path()

xp = cuda.cupy

class RNN(chainer.Chain):
    def __init__(self, params, dataset):
        super(RNN, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(len(dataset.char_id)+1, params["embedding_size"])
            self.bi_lstm = L.NStepBiLSTM(params["LSTM_units"], params["embedding_size"], params["hidden_size"], params["dropout"])
            self.l_app = L.Linear(params["hidden_size"]*2, 1)
            self.l_sep = L.Linear(params["hidden_size"]*2, 1)
            self.wv_embed = L.EmbedID(len(train.word_id)+1, 200, xp.array(train.vectors, xp.float32))
            self.pos_embed = L.EmbedID(len(train.pos_id)+1, 200)
            self.posdetail_embed = L.EmbedID(len(train.posdetail_id)+1, 200)
            self.useful_embed = L.EmbedID(len(train.useful_id)+1, 200)
            self.conjugative_embed = L.EmbedID(len(train.conjugative_id)+1, 200)
        self.n_layer = params["LSTM_units"]
        self.n_units = 150
        self.margin_rate = params["margin_rate"]
        self.dropout = params["dropout"]

    # early_update True ari False nasi
    def __call__(self, xs, ls, early_update=False):
        # make vector
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = F.dropout(self.embed(F.concat(xs, axis=0)), self.dropout)
        exs = F.split_axis(ex, x_section, 0, force_tuple=True)
        hy, cy, ys = self.bi_lstm(hx=None, cx=None, xs=exs)
        ys = F.concat(ys, axis=0) ### これをしないとNStepBiLSTMはbackward()できなくなる
        ys = F.split_axis(ys, x_section, 0, force_tuple=True)

        ###  make character score
        char_scores = []
        for y in ys:
            char_scores.append(self.make_char_score(y))

        cum_loss = 0
        pred_labels = []
        for x, cs, l in zip(xs, char_scores, ls):
            loss, pred_label  = self.search(x, cs, l, early_update)
            cum_loss += loss
            pred_labels.append(pred_label)
        return cum_loss, pred_labels

    def make_char_score(self, y): #shape=(l, 2, lstmout+action*2)を生成, 2はapp,sepの順
        app_score = self.l_app(y)
        sep_score = self.l_sep(y)
        char_scores = F.concat((app_score, sep_score), axis=1)
        return char_scores

    def search(self, x, cs, l, early_update):
        gold_score = 0
        agenda = [[0, [1]]] #[score, label_list]

        for i in range(1, len(l)-1):
            gold_score = gold_score + cs[i][l[i]]

            beam = []
            for one in agenda:
                if l[i] == 0: #goldはappだった sepにmarginが増える
                    app_margin = 0
                    sep_margin = self.margin_rate
                else:
                    app_margin = self.margin_rate
                    sep_margin = 0
                tmp_label = one[1] + [0]
                beam.append([one[0] + cs[i][0] + app_margin, tmp_label])
                tmp_label = one[1] + [1]
                beam.append([one[0] + cs[i][1] + sep_margin, tmp_label])
                
            beam.sort(key=lambda x: x[0].data, reverse=True)
            agenda = beam[:params["beam_size"]]
            
            
            if early_update == True:
                if agenda[-1][0].data > gold_score.data:
                    break

        pred_score = agenda[0][0]
        pred_label = agenda[0][1] + [1]
        pred_label += [7 for _ in range(len(l)-len(pred_label))]
        xpzero = xp.zeros([], dtype=xp.float32)
        loss = F.max(F.stack([xpzero, pred_score - gold_score]))
        return loss, pred_label

    def _word_vec_score(self, current_word, current_char, select): #単語をselectに合わせて更新し，単語が完成した際には対応したlossを返す
        score = 0
        if(select == 1 and current_word != ''):
            current_word = current_word + str(current_char) + '_'
        if(select == 2 and current_word != ''):
            current_word = current_word + str(current_char) + '_'
            if (current_word in word_dict):
                word_id = word_dict[current_word]
                word_vector = self.wv_embed(xp.array([word_id], dtype=xp.int32))
                score = self.lwv(word_vector)[0][0]
            current_word = ''
        if(select == 0):
            current_word = str(current_char) + '_'
        if(select == 3):
            current_word = str(current_char) + '_'
            if (current_word in word_dict):
                word_id = word_dict[current_word]
                word_vector = self.wv_embed(xp.array([word_id], dtype=xp.int32))
                score = self.lwv(word_vector)[0][0]
            current_word = ''
        return current_word, score

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
        
def eval(data):
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

def recorder(data, epoch, path):
    pred_word_count, gold_word_count, correct_word_count, accuracy = eval(data)
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

def accuracy_count(pred, gold):
    count = 0
    for i in range(len(pred)):
        if pred[i] == gold[i]:
            count += 1
    return count

def demo(text, model):
    char_list = [train.char_id["BOS"]]
    label_list = [1]
    for c in text:
        label_list.append(1)
        if c in train.char_id:
            char_list.append(train.char_id[c])
        else:
            char_list.append(len(train.char_id))
    char_list.append(train.char_id["EOS"])
    label_list.append(1)
    _, label = model(xp.array([char_list], dtype=xp.int32), xp.array([label_list], dtype=xp.int32), early_update=False)
    label[0].pop(0)
    for n, c in enumerate(text):
        if label[0][n] == 1:
            print("/", end="")
        print(c, end="")
    print()


train = data.Dataset(params)
train.newread(train_path, wiki_path=wiki_path)
dict_id, dict_list = data.make_dict(dict_path, train)
valid = (train.data[:500], train.label[:500])
del train.data[:500]
del train.label[:500]

print("data load")

model = RNN(params, train)
cuda.get_device(0).use()
model.to_gpu()
optimizer = optimizers.Adam()
optimizer.setup(model)
print("model made")

#####################
##test
# xs = [xp.asarray(train.data[7], dtype=xp.int32),
#       xp.asarray(train.data[8], dtype=xp.int32),
#       xp.asarray(train.data[9], dtype=xp.int32),
#       xp.asarray(train.data[10], dtype=xp.int32)]
# ls = [train.label[7], train.label[8], train.label[9], train.label[10]]
# model.cleargrads()
# loss, pred_labels = model(xs, ls)
# loss.backward()
# optimizer.update()
# for i in range(100):
#     model.cleargrads()
#     loss, pred_labels = model(xs, ls, 1)
#     loss.backward()
#     optimizer.update()
#     print(loss.data)
#     print("gold----", ls[1])
#     print("pred----", pred_labels[1])

# with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

    
#####################
###small data check
# small_data = (train.data[:10], train.label[:10])
# valid = (train.data[-5:], train.label[-5:])
# for i in range(100):
#     for k in range(len(small_data[0])):
#         x = [xp.asarray(small_data[0][k], dtype=xp.int32)]
#         model.cleargrads()
#         loss, pred_labels = model(x, [small_data[1][k]], True)
#         loss.backward()
#         optimizer.update()
#     print("-----------------------")
#     if (i%10 == 0):
#         data.reverse(x[0], small_data[1][k], train.char_id)
#         data.reverse(x[0], pred_labels[0], train.char_id)
#         recorder(valid, i, sys.argv[1])
#         save_path = sys.argv[2] + "/" + str(i)
#         serializers.save_npz(save_path, model)
#     print("answer", small_data[1][k])
#     print("predic", pred_labels[0])
#     print("loss", loss.data)
    

####################
random.seed(0)
random_list = [ x for x in range(len(train.data)) ]
for epoch in range(40):
    count = 0
    sentence_len = 0
    batch_loss = 0
    for n in range(len(random_list)):
        if n%params["batch_size"] == 0:
            x = [xp.asarray(train.data[i], dtype=xp.int32) for i in range(n,n+10)]
            l = [train.label[i] for i in range(n,n+10)]
            model.cleargrads()
            loss, pred_labels = model(x, l, True)
            loss.backward()
            optimizer.update()
            count += accuracy_count(l[0], pred_labels[0]) - 2
            sentence_len += len(l[0]) - 2
            print("epoch", epoch, n, loss.data, "accuracy", count/sentence_len)
    random.shuffle(random_list)
    print("epoch", epoch, "--------------------------------------------------------------------")
    recorder(valid, epoch, sys.argv[1])
    save_path = sys.argv[2] + "/" + str(epoch)
    serializers.save_npz(save_path, model)
    print("----------------------------------------------------------------------------")

# how to reverse
# data.reverse(x[0], l, train.char_id)

# serializers.save_npz("model name", model)
# serializers.load_npz("model name", model)

## test   使い方は通常の訓練と同じ
# test = data.Dataset(params)
# test.read(test_path, char_id)
# test_list = (test.data[:], test.label[:])
# serializers.load_npz('', model)
# recorder(test_list, 1, sys.argv[1])
