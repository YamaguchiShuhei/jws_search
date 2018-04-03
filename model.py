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

xp = cuda.cupy

class ASVEC_left(chainer.Chain):
    """charscore by appsep vector"""
    def __init__(self, params, dataset):
        super(ASVEC_left, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(len(dataset.char_id)+1, params["embedding_size"])
            self.action_embed = L.EmbedID(2, params["action_size"])
            self.bi_lstm = L.NStepBiLSTM(params["LSTM_units"], params["embedding_size"], params["hidden_size"], params["dropout"])
            self.l1 = L.Linear(params["hidden_size"]*2+params["action_size"], params["prescore_size"])
            self.l2 = L.Linear(params["prescore_size"], 1)
            self.wv_embed = L.EmbedID(len(dataset.word_id)+1, 200, xp.array(dataset.vectors, xp.float32))
            self.pos_embed = L.EmbedID(len(dataset.pos_id)+1, 200)
            self.posdetail_embed = L.EmbedID(len(dataset.posdetail_id)+1, 200)
            self.useful_embed = L.EmbedID(len(dataset.useful_id)+1, 200)
            self.conjugative_embed = L.EmbedID(len(dataset.conjugative_id)+1, 200)
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
        self.action = self.action_embed(xp.asarray([0, 1], dtype=xp.int32))

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
        y_matrix = F.broadcast_to(F.expand_dims(y, axis = 1), (len(y), 2, len(y[0])))
        action_matrix = F.broadcast_to(self.action, (len(y), 2, len(self.action[0])))
        char_vecs = F.concat((y_matrix, action_matrix), axis = 2)
        # calc score
        char_list = F.reshape(char_vecs, (len(char_vecs)*2, len(char_vecs[0][0])))
        char_listscore = self.l2(F.tanh(self.l1(char_list)))
        char_scores = F.reshape(char_listscore, (len(char_vecs), 2))
        return char_scores

    def search(self, x, cs, l, early_update):
        gold_score = 0
        agenda = [[0, [1]]] #[score, label_list]

        for i in range(1, len(l)):
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
        pred_label = agenda[0][1]
        pred_label += [7 for _ in range(len(l)-len(pred_label))]
        xpzero = xp.zeros([], dtype=xp.float32)
        loss = F.max(F.stack([xpzero, pred_score - gold_score]))
        return loss, pred_label


class ASVECWV_left(chainer.Chain):
    """charscore by appsep vector and vordscore by wv"""
    def __init__(self, params, dataset):
        super(ASVECWV_left, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(len(dataset.char_id)+1, params["embedding_size"])
            self.action_embed = L.EmbedID(2, params["action_size"])
            self.bi_lstm = L.NStepBiLSTM(params["LSTM_units"], params["embedding_size"], params["hidden_size"], params["dropout"])
            self.l1 = L.Linear(params["hidden_size"]*2+params["action_size"], params["prescore_size"])
            self.l2 = L.Linear(params["prescore_size"], 1)
            self.lwv = L.Linear(200, 1)
            self.wv_embed = L.EmbedID(len(dataset.word_id)+1, 200, xp.array(dataset.vectors, xp.float32))
            self.pos_embed = L.EmbedID(len(dataset.pos_id)+1, 200)
            self.posdetail_embed = L.EmbedID(len(dataset.posdetail_id)+1, 200)
            self.useful_embed = L.EmbedID(len(dataset.useful_id)+1, 200)
            self.conjugative_embed = L.EmbedID(len(dataset.conjugative_id)+1, 200)
        self.n_layer = params["LSTM_units"]
        self.n_units = 150
        self.margin_rate = params["margin_rate"]
        self.dropout = params["dropout"]
        self.word_id = dataset.word_id


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
        self.action = self.action_embed(xp.asarray([0, 1], dtype=xp.int32))

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
        y_matrix = F.broadcast_to(F.expand_dims(y, axis = 1), (len(y), 2, len(y[0])))
        action_matrix = F.broadcast_to(self.action, (len(y), 2, len(self.action[0])))
        char_vecs = F.concat((y_matrix, action_matrix), axis = 2)
        # calc score
        char_list = F.reshape(char_vecs, (len(char_vecs)*2, len(char_vecs[0][0])))
        char_listscore = self.l2(F.tanh(self.l1(char_list)))
        char_scores = F.reshape(char_listscore, (len(char_vecs), 2))
        return char_scores

    def search(self, x, cs, l, early_update):
        gold_score = 0
        gold_word = tuple([x[0].tolist()])
        agenda = [[0, [1], tuple([x[0].tolist()])]] #[score, label_list, current_word]

        for i in range(1, len(l)):
            if l[i] == 0:
                gold_score = gold_score + cs[i][l[i]]
                gold_word = tuple(list(gold_word) + [x[i].tolist()])
            else:
                gold_score = gold_score + cs[i][l[i]] + self._word_vec_score(gold_word)
                gold_word = tuple([x[i].tolist()])

            beam = []
            for one in agenda:
                if l[i] == 0: #goldはappだった sepにmarginが増える
                    app_margin = 0
                    sep_margin = self.margin_rate
                else:
                    app_margin = self.margin_rate
                    sep_margin = 0
                tmp_label = one[1] + [0]
                beam.append([one[0] + cs[i][0] + app_margin, tmp_label, tuple(list(one[2]) + [x[i].tolist()])])
                tmp_label = one[1] + [1]
                beam.append([one[0] + cs[i][1] + sep_margin + self._word_vec_score(one[2]), tmp_label, tuple([x[i].tolist()])])
                
            beam.sort(key=lambda x: x[0].data, reverse=True)
            agenda = beam[:params["beam_size"]]

            if early_update == True:
                if agenda[-1][0].data > gold_score.data:
                    break

        pred_score = agenda[0][0]
        pred_label = agenda[0][1]
        pred_label += [7 for _ in range(len(l)-len(pred_label))]
        xpzero = xp.zeros([], dtype=xp.float32)
        loss = F.max(F.stack([xpzero, pred_score - gold_score]))
        return loss, pred_label

    def _word_vec_score(self, word):
        """return word_score by word"""
        if word in self.word_id:
            return self.lwv(self.wv_embed(xp.array([self.word_id[word]], dtype=xp.int32)))[0][0]
        else:
            return self.lwv(self.wv_embed(xp.array([len(self.word_id)], dtype=xp.int32)))[0][0]


class ASVECWV_right(chainer.Chain):
    """charscore by appsep vector and vordscore by wv"""
    def __init__(self, params, dataset):
        super(ASVECWV_right, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(len(dataset.char_id)+1, params["embedding_size"])
            self.action_embed = L.EmbedID(2, params["action_size"])
            self.bi_lstm = L.NStepBiLSTM(params["LSTM_units"], params["embedding_size"], params["hidden_size"], params["dropout"])
            self.l1 = L.Linear(params["hidden_size"]*2+params["action_size"], params["prescore_size"])
            self.l2 = L.Linear(params["prescore_size"], 1)
            self.lwv = L.Linear(200, 1)
            self.wv_embed = L.EmbedID(len(dataset.word_id)+1, 200, xp.array(dataset.vectors, xp.float32))
            self.pos_embed = L.EmbedID(len(dataset.pos_id)+1, 200)
            self.posdetail_embed = L.EmbedID(len(dataset.posdetail_id)+1, 200)
            self.useful_embed = L.EmbedID(len(dataset.useful_id)+1, 200)
            self.conjugative_embed = L.EmbedID(len(dataset.conjugative_id)+1, 200)
        self.n_layer = params["LSTM_units"]
        self.n_units = 150
        self.margin_rate = params["margin_rate"]
        self.dropout = params["dropout"]
        self.word_id = dataset.word_id


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
        self.action = self.action_embed(xp.asarray([0, 1], dtype=xp.int32))

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
        y_matrix = F.broadcast_to(F.expand_dims(y, axis = 1), (len(y), 2, len(y[0])))
        action_matrix = F.broadcast_to(self.action, (len(y), 2, len(self.action[0])))
        char_vecs = F.concat((y_matrix, action_matrix), axis = 2)
        # calc score
        char_list = F.reshape(char_vecs, (len(char_vecs)*2, len(char_vecs[0][0])))
        char_listscore = self.l2(F.tanh(self.l1(char_list)))
        char_scores = F.reshape(char_listscore, (len(char_vecs), 2))
        return char_scores

    def search(self, x, cs, l, early_update):
        tmp_x = x.tolist() #x をnumpyから listに変換しておく
        tmp_l = l[1:] + [1] #l を一ますずらしてright対応型にしておく
        gold_score = 0
        gold_word = []
        agenda = [[0, [1], []]] #[score, label_list, current_word]
        
        for i in range(1, len(tmp_l) - 1):
            if tmp_l[i] == 0:
                gold_score = gold_score + cs[i][0]
                gold_word = gold_word + [tmp_x[i]]
            else:
                gold_word = gold_word + [tmp_x[i]]
                gold_score = gold_score + cs[i][1] + self._word_vec_score(tuple(gold_word))
                gold_word = []
                
            beam = []
            if tmp_l[i] == 0: #goldはappだった sepにmarginが増える
                app_margin = 0
                sep_margin = self.margin_rate
            else:
                app_margin = self.margin_rate
                sep_margin = 0
            for one in agenda:
                tmp_label = one[1] + [0]
                beam.append([one[0] + cs[i][0] + app_margin, tmp_label, one[2] + [tmp_x[i]]])
                tmp_label = one[1] + [1]
                beam.append([one[0] + cs[i][1] + sep_margin + self._word_vec_score(tuple(one[2] + [tmp_x[i]])), tmp_label, []])
                
            beam.sort(key=lambda x: x[0].data, reverse=True)
            agenda = beam[:params["beam_size"]]

            if early_update == True:
                if agenda[-1][0].data > gold_score.data:
                    break

        pred_score = agenda[0][0]
        pred_label = [1] + agenda[0][1]
        pred_label += [7 for _ in range(len(l)-len(pred_label))]
        xpzero = xp.zeros([], dtype=xp.float32)
        loss = F.max(F.stack([xpzero, pred_score - gold_score]))
        return loss, pred_label

    def _word_vec_score(self, word):
        """return word_score by word"""
        if word in self.word_id:
            return self.lwv(self.wv_embed(xp.array([self.word_id[word]], dtype=xp.int32)))[0][0]
        else:
            return self.lwv(self.wv_embed(xp.array([len(self.word_id)], dtype=xp.int32)))[0][0]

        
class ASVECWV_BMES(chainer.Chain):
    """charscore by appsep vector and vordscore by wv"""
    def __init__(self, params, dataset):
        super(ASVECWV_BMES, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(len(dataset.char_id)+1, params["embedding_size"])
            self.action_embed = L.EmbedID(2, params["action_size"])
            self.bi_lstm = L.NStepBiLSTM(params["LSTM_units"], params["embedding_size"], params["hidden_size"], params["dropout"])
            self.l1 = L.Linear(params["hidden_size"]*2+params["action_size"]*2, params["prescore_size"])
            self.l2 = L.Linear(params["prescore_size"], 1)
            self.lwv = L.Linear(200, 1)
            self.wv_embed = L.EmbedID(len(dataset.word_id)+1, 200, xp.array(dataset.vectors, xp.float32))
            self.pos_embed = L.EmbedID(len(dataset.pos_id)+1, 200)
            self.posdetail_embed = L.EmbedID(len(dataset.posdetail_id)+1, 200)
            self.useful_embed = L.EmbedID(len(dataset.useful_id)+1, 200)
            self.conjugative_embed = L.EmbedID(len(dataset.conjugative_id)+1, 200)
        self.n_layer = params["LSTM_units"]
        self.n_units = 150
        self.margin_rate = params["margin_rate"]
        self.dropout = params["dropout"]
        self.word_id = dataset.word_id

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
        left_boundary = self.action_embed(xp.array([1, 0, 0, 1], dtype=xp.int32)) #BMESの順, つまりaction1がsep
        right_boundary = self.action_embed(xp.array([0, 0, 1, 1], dtype=xp.int32))
        self.boundary = F.concat((left_boundary, right_boundary), axis=1) #action0: app, action1: sep :::: boundary0: B, boundary1: M, boundary2: E, boundary3: S
        
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

    def make_char_score(self, y): #shape=(l, 4, lstmout+action*2)を生成, 4はbmesの順
        y_matrix = F.broadcast_to(F.expand_dims(y, axis = 1), (len(y), 4, len(y[0]))) #shape[sentlen, BMES, hidden(300)]
        boundary_matrix = F.broadcast_to(self.boundary, (len(y), 4, len(self.boundary[0]))) #shape[sentlen, BMES, 100]
        char_vecs = F.concat((y_matrix, boundary_matrix), axis = 2)
        # calc score
        char_list = F.reshape(char_vecs, (len(char_vecs)*4, len(char_vecs[0][0])))
        char_listscore = self.l2(F.tanh(self.l1(char_list)))
        char_scores = F.reshape(char_listscore, (len(char_vecs), 4))
        return char_scores #shape[sentlen, BMES] スカラー

    def search(self, x, cs, l, early_update):
        tmp_x = x.tolist() #x をnumpyから listに変換しておく
        gold_score = 0
        gold_word = []
        agenda = [[0, [1], [], 3]] #[score, label_list, current_word, pred_select]
        
        for i in range(1, len(l) - 1):
            select = self._gold_select(l, i)
            if select in [0, 1]:
                gold_score = gold_score + cs[i][0]
                gold_word = gold_word + [tmp_x[i]]
            else:
                gold_word = gold_word + [tmp_x[i]]
                gold_score = gold_score + cs[i][1] + self._word_vec_score(tuple(gold_word))
                gold_word = []
                
            beam = []
            # if tmp_l[i] == 0: #goldはappだった sepにmarginが増える
            #     app_margin = 0
            #     sep_margin = self.margin_rate
            # else:
            #     app_margin = self.margin_rate
            #     sep_margin = 0
            # for one in agenda:
            #     tmp_label = one[1] + [0]
            #     beam.append([one[0] + cs[i][0] + app_margin, tmp_label, one[2] + [tmp_x[i]]])
            #     tmp_label = one[1] + [1]
            #     beam.append([one[0] + cs[i][1] + sep_margin + self._word_vec_score(tuple(one[2] + [tmp_x[i]])), tmp_label, []])
            for one in agenda:
                if one[3] in [2, 3]: #前がe,sのため、次がb,s
                    tmp_label = one[1] + [1] #b
                    beam.append([one[0] + cs[i][0] + (select != 0) * self.margin_rate, tmp_label, [tmp_x[i]], 0])
                    tmp_label = one[1] + [1] #s
                    beam.append([one[0] + cs[i][3] + (select != 3) * self.margin_rate + self._word_vec_score(tuple(one[2] + [tmp_x[i]])), tmp_label, [], 3])
                else: #前がb,m
                    tmp_label = one[1] + [0] #m
                    beam.append([one[0] + cs[i][1] + (select != 1) * self.margin_rate, tmp_label, one[2] + [tmp_x[i]], 1])
                    tmp_label = one[1] + [0] #e
                    beam.append([one[0] + cs[i][2] + (select != 2) * self.margin_rate + self._word_vec_score(tuple(one[2] + [tmp_x[i]])), tmp_label, [], 2])
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

    def _gold_select(self, l, i): #bmes順
        if l[i] == 1 and l[i+1] == 0:
            return 0
        if l[i] == 0 and l[i+1] == 0:
            return 1
        if l[i] == 0 and l[i+1] == 1:
            return 2
        if l[i] == 1 and l[i+1] == 1:
            return 3
        
    def _word_vec_score(self, word):
        """return word_score by word"""
        if word in self.word_id:
            return self.lwv(self.wv_embed(xp.array([self.word_id[word]], dtype=xp.int32)))[0][0]
        else:
            return self.lwv(self.wv_embed(xp.array([len(self.word_id)], dtype=xp.int32)))[0][0]
        
        
class ASVECRNNWV_right(chainer.Chain):
    """charscore by appsep vector and vordscore by wv"""
    def __init__(self, params, dataset):
        super(ASVECRNNWV_right, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(len(dataset.char_id)+1, params["embedding_size"])
            self.action_embed = L.EmbedID(2, params["action_size"])
            self.bi_lstm = L.NStepBiLSTM(params["LSTM_units"], params["embedding_size"], params["hidden_size"], params["dropout"])
            self.l1 = L.Linear(params["hidden_size"]*2+params["action_size"], params["prescore_size"])
            self.l2 = L.Linear(params["prescore_size"], 1)
            self.lwv = L.Linear(200, 1)
            self.wv_embed = L.EmbedID(len(dataset.word_id)+1, 200, xp.array(dataset.vectors, xp.float32))
            self.wv_lstm = L.NStepLSTM(params["LSTM_units"], 200, 200, params["dropout"])
            # self.pos_embed = L.EmbedID(len(dataset.pos_id)+1, 200)
            # self.posdetail_embed = L.EmbedID(len(dataset.posdetail_id)+1, 200)
            # self.useful_embed = L.EmbedID(len(dataset.useful_id)+1, 200)
            # self.conjugative_embed = L.EmbedID(len(dataset.conjugative_id)+1, 200)
        self.n_layer = params["LSTM_units"]
        self.n_units = 150
        self.margin_rate = params["margin_rate"]
        self.dropout = params["dropout"]
        self.word_id = dataset.word_id


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
        self.action = self.action_embed(xp.asarray([0, 1], dtype=xp.int32))

        ###  make character score
        char_scores = []
        for y in ys:
            char_scores.append(self.make_char_score(y))

        cum_loss = 0
        pred_labels = []
        for x, cs, l in zip(xs, char_scores, ls):
            loss, pred_label = self.search(x, cs, l, early_update)
            cum_loss += loss
            pred_labels.append(pred_label)
        return cum_loss, pred_labels

    def make_char_score(self, y): #shape=(l, 2, lstmout+action*2)を生成, 2はapp,sepの順
        y_matrix = F.broadcast_to(F.expand_dims(y, axis = 1), (len(y), 2, len(y[0])))
        action_matrix = F.broadcast_to(self.action, (len(y), 2, len(self.action[0])))
        char_vecs = F.concat((y_matrix, action_matrix), axis = 2)
        # calc score
        char_list = F.reshape(char_vecs, (len(char_vecs)*2, len(char_vecs[0][0])))
        char_listscore = self.l2(F.tanh(self.l1(char_list)))
        char_scores = F.reshape(char_listscore, (len(char_vecs), 2))
        return char_scores

    def search(self, x, cs, l, early_update):
        tmp_x = x.tolist() #x をnumpyから listに変換しておく
        tmp_l = l[1:] + [1] #l を一ますずらしてright対応型にしておく
        gold_score = 0
        gold_word = [[tmp_x[0]], []]
        agenda = [ [0, [1], [[tmp_x[0]], []]] ] #[score, label_list, pred_word]
        
        for i in range(1, len(tmp_l) - 1):
            words_list = [] # 1文字探索するうちに生じる単語は全部一括でword_scoreを出してしまう
            gold_word[-1] += [tmp_x[i]] #sepでもappでも末尾に文字は確実に足す
            if tmp_l[i] == 0:
                gold_score = gold_score + cs[i][0]
            else:
                gold_score = gold_score + cs[i][1]# + self._word_vec_score(tuple(gold_word))
                words_list.append(gold_word)
                gold_word.append([])
                
            beam = []
            if tmp_l[i] == 0: #goldはappだった sepにmarginが増える
                app_margin = 0
                sep_margin = self.margin_rate
            else:
                app_margin = self.margin_rate
                sep_margin = 0
            for one in agenda:# 先にsepの探索を行う listの参照渡しに気をつけて、しかし重複は避けたいな
                tmp_label = one[1] + [1]
                tmp_word = one[2][:-1] + [one[2][-1] + [tmp_x[i]]]
                words_list.append(tmp_word)
                tmp_word.append([])
                beam.append([one[0] + cs[i][1] + sep_margin, tmp_label, tmp_word])
            # この後wordscoreの足しが入る

            j = 0
            for k, word_score in enumerate(self._word_vec_score(words_list)):
                if k == 0 and sep_margin == 0: #goldの単語スコアを計算している時
                    gold_score += word_score[0]
                    continue
                beam[j][0] += word_score[0]
                j += 1
            
            for one in agenda: #appの探索スペース
                tmp_label = one[1] + [0]
                tmp_word = one[2][:-1] + [one[2][-1] + [tmp_x[i]]]
                beam.append([one[0] + cs[i][0] + app_margin, tmp_label, tmp_word])
                
            beam.sort(key=lambda x: x[0].data, reverse=True)
            agenda = beam[:params["beam_size"]]
            if early_update == True:
                if agenda[-1][0].data > gold_score.data:
                    break

        pred_score = agenda[0][0]
        pred_label = [1] + agenda[0][1]
        pred_label += [7 for _ in range(len(l)-len(pred_label))]
        xpzero = xp.zeros([], dtype=xp.float32)
        loss = F.max(F.stack([xpzero, pred_score - gold_score]))
        return loss, pred_label

    def _word_vec_score(self, words_list):
        """return word_score by word"""
        words_array = [xp.array([self.word_id[tuple(word)] if tuple(word) in self.word_id else len(self.word_id) for word in words], dtype = xp.int32) for words in words_list]
        # make vector
        words_len = [len(words) for words in words_array]
        words_section = np.cumsum(words_len[:-1])
        ewords = F.dropout(self.wv_embed(F.concat(words_array, axis=0)), self.dropout)
        ewords_list = F.split_axis(ewords, words_section, 0, force_tuple=True)
        hy, cy, wys = self.wv_lstm(hx=None, cx=None, xs=ewords_list)
        wys = F.concat(wys, axis=0) ### これをしないとNStepBiLSTMはbackward()できなくなる
        wys = F.split_axis(wys, words_section, 0, force_tuple=True)
        word_vec = F.reshape(F.concat([wy[-1] for wy in wys], axis=0), [-1, 200])
        return self.lwv(word_vec)


class ASVECWVMO_left(chainer.Chain):
    """charscore by appsep vector and wordscore by mo inf"""
    def __init__(self, params, dataset):
        super(APVECWVMO, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(len(dataset.char_id)+1, params["embedding_size"])
            self.action_embed = L.EmbedID(2, params["action_size"])
            self.bi_lstm = L.NStepBiLSTM(params["LSTM_units"], params["embedding_size"], params["hidden_size"], params["dropout"])
            self.l1 = L.Linear(params["hidden_size"]*2+params["action_size"], params["prescore_size"])
            self.l2 = L.Linear(params["prescore_size"], 1)
            self.lwv = L.Linear(200, 1)
            self.wv_embed = L.EmbedID(len(dataset.word_id)+1, 200, xp.array(dataset.vectors, xp.float32))
            self.pos_embed = L.EmbedID(len(dataset.pos_id)+1, 200)
            self.posdetail_embed = L.EmbedID(len(dataset.posdetail_id)+1, 200)
            self.useful_embed = L.EmbedID(len(dataset.useful_id)+1, 200)
            self.conjugative_embed = L.EmbedID(len(dataset.conjugative_id)+1, 200)
        self.n_layer = params["LSTM_units"]
        self.n_units = 150
        self.margin_rate = params["margin_rate"]
        self.dropout = params["dropout"]
        self.word_id = dataset.word_id
        self.dict_id = dataset.dict_id
        self.dict_list = dataset.dict_list

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
        self.action = self.action_embed(xp.asarray([0, 1], dtype=xp.int32))

        ###  make character score
        char_scores = []
        for y in ys:
            char_scores.append(self.make_char_score(y))

        ### start searching
        cum_loss = 0
        pred_labels = []
        for x, cs, l in zip(xs, char_scores, ls):
            loss, pred_label  = self.search(x, cs, l, early_update)
            cum_loss += loss
            pred_labels.append(pred_label)
        return cum_loss, pred_labels

    def make_char_score(self, y): #shape=(l, 2, lstmout+action*2)を生成, 2はapp,sepの順
        y_matrix = F.broadcast_to(F.expand_dims(y, axis = 1), (len(y), 2, len(y[0])))
        action_matrix = F.broadcast_to(self.action, (len(y), 2, len(self.action[0])))
        char_vecs = F.concat((y_matrix, action_matrix), axis = 2)
        # calc score
        char_list = F.reshape(char_vecs, (len(char_vecs)*2, len(char_vecs[0][0])))
        char_listscore = self.l2(F.tanh(self.l1(char_list)))
        char_scores = F.reshape(char_listscore, (len(char_vecs), 2))
        return char_scores

    def search(self, x, cs, l, early_update):
        gold_score = 0
        gold_word = tuple([x[0].tolist()])
        agenda = [[0, [1], tuple([x[0].tolist()])]] #[score, label_list, current_word]

        for i in range(1, len(l)):
            if l[i] == 0:
                gold_score = gold_score + cs[i][l[i]]
                gold_word = tuple(list(gold_word) + [x[i].tolist()])
            else:
                gold_score = gold_score + cs[i][l[i]] + self._word_vec_score(gold_word)
                gold_word = tuple([x[i].tolist()])

            beam = []
            for one in agenda:
                if l[i] == 0: #goldはappだった sepにmarginが増える
                    app_margin = 0
                    sep_margin = self.margin_rate
                else:
                    app_margin = self.margin_rate
                    sep_margin = 0
                tmp_label = one[1] + [0]
                beam.append([one[0] + cs[i][0] + app_margin, tmp_label, tuple(list(one[2]) + [x[i].tolist()])])
                tmp_label = one[1] + [1]
                beam.append([one[0] + cs[i][1] + sep_margin + self._word_vec_score(one[2]), tmp_label, tuple([x[i].tolist()])])
                
            beam.sort(key=lambda x: x[0].data, reverse=True)
            agenda = beam[:params["beam_size"]]

            if early_update == True:
                if agenda[-1][0].data > gold_score.data:
                    break

        pred_score = agenda[0][0]
        pred_label = agenda[0][1]
        pred_label += [7 for _ in range(len(l)-len(pred_label))]
        xpzero = xp.zeros([], dtype=xp.float32)
        loss = F.max(F.stack([xpzero, pred_score - gold_score]))
        return loss, pred_label

    def _word_vec_score(self, word):
        """return word_score by word"""
        if word in self.dict_id:
            tmp = self.dict_list[self.dict_id[word]].push()
            word_id_emb = F.expand_dims(self.wv_embed(xp.array(tmp[0], xp.int32)), 1)
            pos_id_emb = F.expand_dims(self.pos_embed(xp.array(tmp[1], xp.int32)), 1)
            posdetail_id_embed = F.expand_dims(self.posdetail_embed(xp.array(tmp[2], xp.int32)), 1)
            useful_id_embed = F.expand_dims(self.useful_embed(xp.array(tmp[3], xp.int32)), 1)
            conjugative_id_embed = F.expand_dims(self.conjugative_embed(xp.array(tmp[4], xp.int32)), 1)
            dict_embed = F.average(F.concat([word_id_emb, pos_id_emb, posdetail_id_embed, useful_id_embed, conjugative_id_embed], 1), 1)
        elif word in self.word_id:
            word_id_emb = self.wv_embed(xp.array([self.word_id[word]], xp.int32))
            pos_id_emb = self.pos_embed(xp.array([len(self.pos_embed.W) - 1], xp.int32))
            posdetail_id_embed = self.posdetail_embed(xp.array([len(self.posdetail_embed.W) - 1], xp.int32))
            useful_id_embed = self.useful_embed(xp.array([len(self.useful_embed.W) - 1], xp.int32))
            conjugative_id_embed = self.conjugative_embed(xp.array([len(self.conjugative_embed.W) - 1], xp.int32))
            dict_embed = F.expand_dims(F.average(F.concat([word_id_emb, pos_id_emb, posdetail_id_embed, useful_id_embed, conjugative_id_embed], 0), 0), 0)
        else:
            word_id_emb = self.wv_embed(xp.array([len(self.wv_embed.W) - 1], xp.int32))
            pos_id_emb = self.pos_embed(xp.array([len(self.pos_embed.W) - 1], xp.int32))
            posdetail_id_embed = self.posdetail_embed(xp.array([len(self.posdetail_embed.W) - 1], xp.int32))
            useful_id_embed = self.useful_embed(xp.array([len(self.useful_embed.W) - 1], xp.int32))
            conjugative_id_embed = self.conjugative_embed(xp.array([len(self.conjugative_embed.W) - 1], xp.int32))
            dict_embed = F.expand_dims(F.average(F.concat([word_id_emb, pos_id_emb, posdetail_id_embed, useful_id_embed, conjugative_id_embed], 0), 0), 0)
        return dict_embed
    

class ASLIN_left(chainer.Chain):
    """charscore by appsep linear"""
    def __init__(self, params, dataset):
        super(ASLIN_left, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(len(dataset.char_id)+1, params["embedding_size"])
            self.bi_lstm = L.NStepBiLSTM(params["LSTM_units"], params["embedding_size"], params["hidden_size"], params["dropout"])
            self.l_app = L.Linear(params["hidden_size"]*2, 1)
            self.l_sep = L.Linear(params["hidden_size"]*2, 1)
            self.wv_embed = L.EmbedID(len(dataset.word_id)+1, 200, xp.array(dataset.vectors, xp.float32))
            self.pos_embed = L.EmbedID(len(dataset.pos_id)+1, 200)
            self.posdetail_embed = L.EmbedID(len(dataset.posdetail_id)+1, 200)
            self.useful_embed = L.EmbedID(len(dataset.useful_id)+1, 200)
            self.conjugative_embed = L.EmbedID(len(dataset.conjugative_id)+1, 200)
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

        for i in range(1, len(l)):
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
        pred_label = agenda[0][1]
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
