class Dial:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        self.glove_voc = {}
        self.glove_vec = None

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class BClass:
    def __init__(self, num, max_score):
        self.n_class = num
        self.n_max_score = max_score
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addScoreList(self, score_list):
        for score in score_list:
            self.addWord(score)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
