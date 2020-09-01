'''
universal tag set used
'''

from nltk.corpus import brown as training_data
import nltk
from collections import defaultdict
from math import log
class Viterbi:
    unique_tags_list = ['DET', 'NOUN', 'ADJ', 'VERB', 'ADP', '.', 'ADV', 'CONJ', 'PRT', 'PRON', 'NUM', 'X']
    def __init__(self, tagged_sents_training_data):
        self.tagged_sents_training_data=tagged_sents_training_data
        self.tag_count_dict=defaultdict(int)
        self.word_count_dict=defaultdict(int)
        self.word_tag_count_dictionary=defaultdict(int)
        self.tag_tag_count_dictionary=defaultdict(int)
        self.unique_word_list=[]
        self.observation_table=defaultdict(int)
        self.transition_table=defaultdict(int)
        self.log_base=2
        self.dataSetup()
        self.buildObservationTable()
        self.buildTransitionTable()

    def dataSetup(self):
        for sentence in self.tagged_sents_training_data:
            tag_list = ['^']
            for (word, tag) in sentence:
                self.tag_count_dict[tag] += 1
                self.word_count_dict[word] += 1
                self.word_tag_count_dictionary[(word, tag)] += 1
                tag_list.append(tag)
            tag_bigrams = list(nltk.bigrams(tag_list))
            # tag_bigrams = list(nltk.bigrams([tag for (word, tag) in training_data_tagged_words_set1]))
            for (pre_tag, post_tag) in tag_bigrams:
                self.tag_tag_count_dictionary[(pre_tag, post_tag)] += 1
        self.unique_word_list = list(self.word_count_dict.keys())

    def buildObservationTable(self):
        unique_tag_list_temp=['^']+self.unique_tags_list
        for tag in unique_tag_list_temp:
            for word in self.unique_word_list:
                self.observation_table[(word,tag)]=(self.word_tag_count_dictionary[(word,tag)]+1)/(self.tag_count_dict[tag]+len(self.unique_word_list))

    def buildTransitionTable(self):
        unique_tag_list_temp = ['^'] + self.unique_tags_list
        for pre_tag in unique_tag_list_temp:
            for post_tag in unique_tag_list_temp:
                self.transition_table[(pre_tag, post_tag)]=(self.tag_tag_count_dictionary[(pre_tag, post_tag)]+1)/(self.tag_count_dict[pre_tag]+len(unique_tag_list_temp))
    def execute(self, sentence):
        tags = ['^'] + self.unique_tags_list
        testStringLegth = len(sentence)
        tagLength = len(tags)
        dp = [[None] * testStringLegth for _ in range(tagLength)]
        for i in range(1, tagLength):
            if (self.observation_table[(sentence[0], tags[i])] == 0):
                self.observation_table[(sentence[0], tags[i])] = 1 / (self.tag_count_dict[tags[i]] + len(self.unique_word_list))
            p = log(self.observation_table[(sentence[0], tags[i])] * self.transition_table[('^', tags[i])], self.log_base)
            dp[i][0] = (p, 0);
        for j in range(1, testStringLegth):
            for i in range(1, tagLength):
                max, prev = float('-inf'), 0
                if (self.observation_table[(sentence[j], tags[i])] == 0):
                    self.observation_table[(sentence[j], tags[i])] = 1 / (self.tag_count_dict[tags[i]] + len(self.unique_word_list))
                for k in range(1, tagLength):
                    p = dp[k][j - 1][0] + log(self.transition_table[(tags[k], tags[i])], self.log_base) + log(
                        self.observation_table[(sentence[j], tags[i])], self.log_base)
                    if (p != float('-inf') and p > max):
                        max = p
                        prev = k
                dp[i][j] = (max, prev)

        # find the tag sequence
        max, ind = float('-inf'), 0
        tag_output_sequence = [None] * testStringLegth
        j = testStringLegth - 1
        for i in range(1, tagLength):
            if (dp[i][j][0] > max):
                max, ind = dp[i][j]
        tag_output_sequence[j] = '.'
        while (tags[ind] != '^'):
            tag_output_sequence[j - 1] = tags[ind]
            ind = dp[ind][j - 1][1]
            j -= 1
        return tag_output_sequence
