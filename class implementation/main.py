'''
Mistakes:
1. while initializing 2D array don't create copies of rows otherwise any update
in any row will affect all the rows
2. my training data contain (rain, NOUN) if raining comes is test data
it will give complete tag sequence as null because probabilites are muliplying -->> smoothing
3.
'''
from nltk.corpus import brown as training_data
import nltk
from collections import defaultdict
from tabulate import tabulate
import sys
sys.path.append(".")
from FirstOrderHMMViterbi import Viterbi

expected_predicted_dict=defaultdict(int);
confusion_matrix=defaultdict(int);
unique_tag_set_list=Viterbi.unique_tags_list
tag_set_length=len(unique_tag_set_list)
splitFile=100
for test_iteration in range(5):
    tests=test_iteration*100
    testf=test_iteration*100+100
    log_base=2
    fileIds = training_data.fileids()
    training_fileIds_set1 = fileIds[:tests]+fileIds[testf:]
    test_fileIds_set1 = fileIds[tests:testf]
    training_data_tagged_sents_set1=training_data.tagged_sents(fileids=training_fileIds_set1, tagset='universal')
    test_data_sents_set1=training_data.sents(fileids=test_fileIds_set1)
    expected_data_sents_set1=training_data.tagged_sents(fileids=test_fileIds_set1, tagset='universal')
    viterbi = Viterbi(training_data_tagged_sents_set1)
    correct_tag_count=0
    total_tag_count=0
    for ind, test_sents in enumerate(test_data_sents_set1):

        actual_tag_seq=[tag for (word,tag) in expected_data_sents_set1[ind]]
        predicted_tag_seq=viterbi.execute(test_sents)
        for i in range(len(actual_tag_seq)):
            if(actual_tag_seq[i]==predicted_tag_seq[i]):
                correct_tag_count+=1
            total_tag_count+=1
            expected_predicted_dict[(actual_tag_seq[i],predicted_tag_seq[i])]+=1
    print("accuracy in iteration%d: %d%c"%(test_iteration+1,(correct_tag_count*100.0/total_tag_count),37))

#calculate confusion matrix
confusion_matrix=[[0]*(tag_set_length+1) for _ in range(tag_set_length+1)]
per_tag_accuracy_matrix=[[0]*(tag_set_length+1) for _ in range(2)]
tag_index_dict=defaultdict(int)
for index, tag in enumerate(unique_tag_set_list):
    tag_index_dict[tag]=index
#rows actual tag #columns are predicted tags
confusion_matrix[0][0]='actual tag on rows'
per_tag_accuracy_matrix[0][0]='Tag'
per_tag_accuracy_matrix[1][0]='Accuracy'
for i in range(1,tag_set_length+1):
    confusion_matrix[i][0]=unique_tag_set_list[i-1]
    confusion_matrix[0][i]=unique_tag_set_list[i-1]
    per_tag_accuracy_matrix[0][i]=unique_tag_set_list[i-1]

for (actual_tag, predicted_tag), value in expected_predicted_dict.items():
    confusion_matrix[tag_index_dict[actual_tag]+1][tag_index_dict[predicted_tag]+1]+=value

#per tag accuracy calculation

for i in range(1,tag_set_length+1):
    sum=0
    for j in range(1,tag_set_length+1):
        sum+=confusion_matrix[i][j]
    per_tag_accuracy_matrix[1][i]=format(confusion_matrix[i][i]/sum*1.0,'.4f')

print(tabulate(confusion_matrix, tablefmt="fancy_grid"))
print(tabulate(per_tag_accuracy_matrix, tablefmt="fancy_grid"))
