'''
Mistakes:
1. while initializing 2D array don't create copies of rows otherwise any update
in any row will affect all the rows
2. my training data contain (rain, NOUN) if raining comes is test data
it will give complete tag sequence as null because probabilites are muliplying -->> smoothing
3.
'''
from nltk.corpus import brown as training_data
import pandas as pd
import nltk
from math import log
from math import exp
from collections import defaultdict

def setupData(tagged_sents_training_data):
    for sentence in tagged_sents_training_data:
        tag_list = ['^']
        for (word, tag) in sentence:
            tag_count_dict[tag] += 1
            word_count_dict[word] += 1
            word_tag_count_dictionary[(word, tag)] += 1
            tag_list.append(tag)
        tag_bigrams = list(nltk.bigrams(tag_list))
        # tag_bigrams = list(nltk.bigrams([tag for (word, tag) in training_data_tagged_words_set1]))
        for (pre_tag, post_tag) in tag_bigrams:
            tag_tag_count_dictionary[(pre_tag, post_tag)] += 1
        tag_count_dict['^'] += 1

def buildObservationTable(unique_tag_set, unique_words):
    for tag in unique_tag_set:
        for word in unique_words:
            observation_table[(word,tag)]=(word_tag_count_dictionary[(word,tag)]+1)/(tag_count_dict[tag]+len(unique_words))

def buildTransitionTable(unique_tags):
    for pre_tag in unique_tags:
        for post_tag in unique_tags:
            transition_table[(pre_tag, post_tag)]=(tag_tag_count_dictionary[(pre_tag, post_tag)]+1)/(tag_count_dict[pre_tag]+len(unique_tags))

#a-->> testString ,, b --->> unique_tag_list
def ViterbiHMMFirstOrder(a, b ):
    testStringLegth=len(a)
    tagLength=len(b)
    dp=[[None]*testStringLegth for _ in range(tagLength)]
    for i in range(1,tagLength):
        if(observation_table[(a[0],b[i])]==0):
            observation_table[(a[0], b[i])]=1/(tag_count_dict[b[i]]+len(unique_word_set_list))
        p=log(observation_table[(a[0],b[i])]*transition_table[('^',b[i])],log_base)
        dp[i][0]=(p,0);
    for j in range(1,testStringLegth):
        for i in range(1,tagLength):
            max, prev = float('-inf'), 0
            if (observation_table[(a[j],b[i])] == 0):
                observation_table[(a[j], b[i])] = 1 / (tag_count_dict[b[i]] + len(unique_word_set_list))
            for k in range(1,tagLength):
                p=dp[k][j-1][0]+log(transition_table[(b[k],b[i])],log_base)+log(observation_table[(a[j],b[i])],log_base)
                if(p!=float('-inf') and p>max):
                    max=p
                    prev=k
            dp[i][j]=(max,prev)
    #print(dp)

    #find the tag sequence
    max, ind=float('-inf'),0
    tag_output_sequence = [None] * testStringLegth
    j = testStringLegth - 1
    for i in range(1,tagLength):
        if(dp[i][j][0]>max):
            max,ind=dp[i][j]
    tag_output_sequence[j] = '.'
    while(unique_tag_set_list[ind]!='^'):
        tag_output_sequence[j-1]=unique_tag_set_list[ind]
        ind=dp[ind][j-1][1]
        j-=1
    return tag_output_sequence




splitFile=100
for test_iteration in range(5):
    tests=test_iteration*100
    testf=test_iteration*100+100
    log_base=2

    #variable setup
    fileIds = training_data.fileids()
    training_fileIds_set1 = fileIds[:tests]+fileIds[testf:]
    test_fileIds_set1 = fileIds[tests:testf]
    training_data_tagged_sents_set1=training_data.tagged_sents(fileids=training_fileIds_set1, tagset='universal')
    test_data_sents_set1=training_data.sents(fileids=test_fileIds_set1)
    expected_data_sents_set1=training_data.tagged_sents(fileids=test_fileIds_set1, tagset='universal')

    tag_count_dict=defaultdict(int);
    word_count_dict=defaultdict(int);
    word_tag_count_dictionary=defaultdict(int);
    tag_tag_count_dictionary=defaultdict(int);
    observation_table=defaultdict(int);
    transition_table=defaultdict(int);


    #extract useful information from corpus
    setupData(training_data_tagged_sents_set1)
    unique_tag_set_list=list(tag_count_dict.keys())
    unique_word_set_list=list(word_count_dict.keys())
    unique_tag_set_list.insert(0,'^')

    #build observation table and transition table
    buildObservationTable(unique_tag_set_list, unique_word_set_list)
    buildTransitionTable(unique_tag_set_list)


    #execute viterbi and calculate percentage of accuracy
    correct_tag_count=0
    total_tag_count=0
    for ind, test_sents in enumerate(test_data_sents_set1):
        #print(test_sents)
        expected_tag_seq=[tag for (word,tag) in expected_data_sents_set1[ind]]
        #print(expected_tag_seq)
        output_tag_seq=ViterbiHMMFirstOrder(test_sents,unique_tag_set_list)
        #print(output_tag_seq)
        for i in range(len(expected_tag_seq)):
            if(expected_tag_seq[i]==output_tag_seq[i]):
                correct_tag_count+=1
            total_tag_count+=1
    print("accuracy in iteration%d: %d%c"%(test_iteration+1,(correct_tag_count*100.0/total_tag_count),37))
