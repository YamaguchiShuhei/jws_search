import collections
# word head char = 1 , word other char= 0
import pickle
import time
import numpy as np

params = {"FREQ_times":3, "SENT_len":300, "embedding_size":100, "batch_size":10, "CHAR_size":4000,"hidden_size":150, 'LSTM_units':1, "dropout":0.2, "margin_rate":0.2}

class Dataset:
    def __init__(self, params):
        self.data = None
        self.label = None
        self.word_label = None
        self.pos_label = None
        self.posdetail_label = None
        self.useful_label = None
        self.conjugative_label = None
        self.char_id = None
        self.word_id = None
        self.pos_id = {"*":0}
        self.posdetail_id = {"*":0}
        self.useful_id = {"*":0}
        self.conjugative_id = {"*":0}
        self.vectors = None
        self.params = params

    def _read_charword(self, data_path):
        """BOSとEOSを含んだ文字だけのリストと単語の集合を作るだけ"""
        char_list = ["BOS"]
        firstword_set = set()
        word_set = set()
        for x in open(data_path):
            word = x.strip().split()[0]
            if word == "EOS":
                char_list.append(word)
                char_list.append("BOS")
            else:
                char_list.extend(word)
                if word in firstword_set:
                    word_set.add(word)
                else:
                    firstword_set.add(word)
        char_list.pop()
        return char_list, word_set
        
    def _character_id(self, char_list):
        counter = collections.Counter(char_list)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        count_pairs_highfreq = [ i for i in count_pairs if i[1] > self.params["FREQ_times"] ]
        words, _ = list(zip(*count_pairs_highfreq))
        char_id = dict(zip(words, range(len(words))))
        return char_id
    
    def _wikiword_id(self, word_set, wiki_path, char_id):
        """wikivectorを考慮したword_idを作る"""
        word_id = {}
        vectors = []
        for line in open(wiki_path):
            tmp = line.strip().split()
            if tmp[0] in word_set:
                word_set.remove(tmp[0])
                tup = word2tuple(tmp[0], char_id)
                if tup not in word_id: #vectorの方の前処理による重複表層を避けるため
                    word_id[tup] = len(word_id)
                    vectors.append(list(map(float, tmp[1:])))
        return word_id, vectors, word_set
                
    def _word_id(self, word_set, char_id, word_id=None, vectors=None):
        """wikivectorを考慮していないword_idを作る"""
        if word_id == None:
            word_id = {}
            vectors = []
        word_id[tuple([char_id["BOS"]])] = len(word_id) ###BOSとeosをword_idに加えてあげる
        word_id[tuple([char_id["EOS"]])] = len(word_id)
        vectors.append(list(map(float, np.random.uniform(-5, 5, 200).tolist())))
        vectors.append(list(map(float, np.random.uniform(-5, 5, 200).tolist())))
        for word in word_set:
            tup = word2tuple(word, char_id)
            word_id[tup] = len(word_id)
            vectors.append(list(map(float, np.random.uniform(-5, 5, 200).tolist())))
        vectors.append(list(map(float, np.random.uniform(-5, 5, 200).tolist()))) ##未知語用のvector
        return word_id, vectors

    def _make_sentence(self, text_path):
        sentence_list = []
        sentence_label_list = []
        sentence_word_list = []
        sentence_pos_list = []
        sentence_detail_list = []
        sentence_useful_list = []
        sentence_conjugative_list = []
        sent = [self.char_id["BOS"]]
        label = [1]
        word = [self.word_id[tuple([self.char_id["BOS"]])]]
        pos = [self.pos_id["*"]]
        detail = [self.posdetail_id["*"]]
        useful = [self.useful_id["*"]]
        conjugative = [self.conjugative_id["*"]]
        for line in open(text_path):
            tmp = line.strip().split()
            if tmp[0] == "EOS":
                sent.append(self.char_id["EOS"])
                label.append(1)
                word.append(self.word_id[tuple([self.char_id["EOS"]])])
                pos.append(self.pos_id["*"])
                detail.append(self.posdetail_id["*"])
                useful.append(self.useful_id["*"])
                conjugative.append(self.conjugative_id["*"])
                sentence_list.append(sent)
                sentence_label_list.append(label)
                sentence_word_list.append(word)
                sentence_pos_list.append(pos)
                sentence_detail_list.append(detail)
                sentence_useful_list.append(useful)
                sentence_conjugative_list.append(conjugative)
                sent = [self.char_id["BOS"]]
                label = [1]
                word = [self.word_id[tuple([self.char_id["BOS"]])]]
                pos = [self.pos_id["*"]]
                detail = [self.posdetail_id["*"]]
                useful = [self.useful_id["*"]]
                conjugative = [self.conjugative_id["*"]]
            else:
                for i, c in enumerate(tmp[0]):
                    if i == 0:
                        label.append(1)
                    else:
                        label.append(0)
                    if c in self.char_id:
                        sent.append(self.char_id[c])
                    else:
                        sent.append(len(self.char_id))
                    tup = word2tuple(tmp[0], self.char_id)
                    if tup in self.word_id:
                        word.append(self.word_id[tup])
                    else:
                        word.append(len(self.word_id))
                    if tmp[3] in self.pos_id:
                        pos.append(self.pos_id[tmp[3]])
                    else:
                        self.pos_id[tmp[3]] = len(self.pos_id)
                        pos.append(self.pos_id[tmp[3]])
                    if tmp[4] in self.posdetail_id:
                        detail.append(self.posdetail_id[tmp[4]])
                    else:
                        self.posdetail_id[tmp[4]] = len(self.posdetail_id)
                        detail.append(self.posdetail_id[tmp[4]])
                    if tmp[5] in self.useful_id:
                        useful.append(self.useful_id[tmp[5]])
                    else:
                        self.useful_id[tmp[5]] = len(self.useful_id)
                        useful.append(self.useful_id[tmp[5]])
                    if tmp[6] in self.conjugative_id:
                        conjugative.append(self.conjugative_id[tmp[6]])
                    else:
                        self.conjugative_id[tmp[6]] = len(self.conjugative_id)
                        conjugative.append(self.conjugative_id[tmp[6]])
        
        return sentence_list, sentence_label_list, sentence_word_list, sentence_pos_list, sentence_detail_list, sentence_useful_list, sentence_conjugative_list

    def newread(self, data_path, char_id=None, wiki_path=None):
        """ 生文からchar_id, wikiからword_idを作る """
        start = time.time()
        self.char_list, word_set = self._read_charword(data_path)
        if char_id != None:
            self.char_id = char_id
        else:
            self.char_id = self._character_id(self.char_list)
        if wiki_path != None:
            self.word_id, self.vectors, word_set = self._wikiword_id(word_set, wiki_path, self.char_id)
            self.word_id, self.vectors = self._word_id(word_set, self.char_id, word_id=self.word_id, vectors=self.vectors)
        else:
            self.word_id, self.vectors = self._word_id(word_set, self.char_id)
        self.data, self.label, self.word_label, self.pos_label, self.posdetail_label, self.useful_label, self.conjugative_label = self._make_sentence(data_path)
        print(time.time() - start)
    
class Morpheme:
    def __init__(self):
        self.raw = None
        self.line = None
        self.pos = None
        self.posdetail = None
        self.useful = None
        self.conjugative = None

    def read(self, word, dataset):
        self.raw = word[0]
        self.line = word
        if word[4] in dataset.pos_id:
            self.pos = dataset.pos_id[word[4]]
        else:
            self.pos = len(dataset.pos_id)
        if word[5] in dataset.posdetail_id:
            self.posdetail = dataset.posdetail_id[word[5]]
        else:
            self.posdetail = len(dataset.posdetail_id)
        if word[6] in dataset.useful_id:
            self.useful = dataset.useful_id[word[6]]
        else:
            self.useful = len(dataset.useful_id)
        if word[7] in dataset.conjugative_id:
            self.conjugative = dataset.conjugative_id[word[7]]
        else:
            self.conjugative = len(dataset.conjugative_id)

    def oow(self, dataset):
        self.line = "未知語"
        self.pos = len(dataset.pos_id)
        self.posdetail = len(dataset.posdetail_id)
        self.useful = len(dataset.useful_id)
        self.conjugative = len(dataset.conjugative_id)
        

class Word:
    """表層によって管理される単語情報"""
    def __init__(self, dataset):
        self.raw = None
        self.word_id = None
        self.tup = None
        morpheme = Morpheme() ###初回に未知語まで登録してしまう
        morpheme.oow(dataset)
        self.morphemes = [morpheme]
        
    def read(self, word, dataset):
        """表層に対して形態素情報は個別だが、word_idは固有"""
        self.raw = word[0]
        self.tup = word2tuple(word[0], dataset.char_id)
        if self.tup in dataset.word_id:
            self.word_id = dataset.word_id[self.tup]
        else:
            self.word_id = len(dataset.word_id)
        morpheme = Morpheme()
        morpheme.read(word, dataset)
        self.morphemes.append(morpheme)

    def add(self, word, dataset):
        """形態素情報のみの追加"""
        morpheme = Morpheme()
        morpheme.read(word, dataset)
        self.morphemes.append(morpheme)

    def push(self):
        word = []
        pos = []
        posdetail = []
        useful = []
        conjugative = []
        for morpheme in self.morphemes:
            word.append(self.word_id)
            pos.append(morpheme.pos)
            posdetail.append(morpheme.posdetail)
            useful.append(morpheme.useful)
            conjugative.append(morpheme.conjugative)
        return (word, pos, posdetail, useful, conjugative)
        

def make_dict(dict_path, dataset):
    start = time.time()
    dict_id = {}
    dict_list = []
    char_id = dataset.char_id
    word_id = dataset.word_id
    juman = [line.strip().split(",") for line in open(dict_path)]

    for word in juman:
        tup = word2tuple(word[0], char_id)
        if tup not in dict_id:
            dict_id[tup] = len(dict_id)
            tmp = Word(dataset)
            tmp.read(word, dataset)
            dict_list.append(tmp)
        else:
            dict_list[dict_id[tup]].add(word, dataset)
    print(time.time() - start)
    return dict_id, dict_list
    

def word2tuple(word, char_id):
    """word 2 tuple by char_id"""
    tup = []
    for c in word:
        if str(c) in char_id:
            tup.append(char_id[c])
        else:
            tup.append(len(char_id))
    return tuple(tup)
        
def reverse(x, l, char_id):
    for i, w in enumerate(x):
        if l[i] == 1:
            print("/", end="")
        idx2char(w, char_id)
    print()

def idx2char(idx, char_id):
    exist = 0
    for char in char_id:
        if char_id[char] == idx:
            print(char, end="")
            exist+=1
    if exist == 0:
        print("#", end="")

def word_vector_reverse(word, char_id):
    for char in word:
        idx2char(int(char), char_id)
    print()

def entity_load(entity_path, char_id):
    word_vector = []
    word_dict = {}
    word_count = 0

    for x in open(entity_path):
        entity = x.split()
        word = ''
        if (len(entity) != 201):
            continue
        for char in entity[0]:
            if char in char_id:
                word = word + str(char_id[char]) + "_"
            else:
                word = word + str(len(char_id)) + "_"
        if word  not in word_dict: #重複を避けるための措置
            word_vector.append(list(map(float, entity[1:])))
            word_dict.update({word:word_count})
            word_count += 1
    return word_dict, word_vector
        
# with open('char_id.pickle', 'rb') as f:
#     char_id = pickle.load(f)

# word_vector = []
# word_dict = {}
# word_count = 0
# for x in open(entity_path):
#     entity = x.split()
#     word_vector.append(entity[1:])
#     word = ''
#     for char in entity[0]:
#         if char in char_id:
#             word = word + str(char_id[char]) + "_"
#         else:
#             word = word + str(len(char_id)) + "_"
#     word_dict.update({word:word_count})
#     word_count += 1
    
#dd = Dataset(params)
#dd.newread(text_path, wiki_path=wiki_path)
