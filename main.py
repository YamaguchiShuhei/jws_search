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
import model
import record


params = {"FREQ_times":3, "SENT_len":300, "embedding_size":100, "batch_size":10, "CHAR_size":3000,"hidden_size":150, "action_size":50, "prescore_size":300, "wordvec_size":200, "LSTM_units":1, "dropout":0.2, "margin_rate":0.2, "beam_size":8}

train_path, test_path, wiki_path, dict_path = path.path()

xp = cuda.cupy

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
train.dict_id, train.dict_list = data.make_dict(dict_path, train)
valid = (train.data[:500], train.label[:500], train.pos_label[:500], train.posdetail_label[:500], train.useful_label[:500], train.conjugative_label[:500])
del train.data[:500], train.label[:500], train.pos_label[:500], train.posdetail_label[:500], train.useful_label[:500], train.conjugative_label[:500]
print("data load")

model = model.ASVECWV_right(params, train)
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
#         record.recorder(valid, i, sys.argv[1], model)
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
    record.recorder(valid, epoch, sys.argv[1], model)
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
# record.recorder(test_list, 1, sys.argv[1], model)
