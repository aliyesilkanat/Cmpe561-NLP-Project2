
# coding: utf-8

# In[1]:


import random,subprocess
from enum import Enum
from collections import Counter

    
def generate_model_line(s,idx): #generates svm input vectors using predefined character numbers
    feature_list=[]
    raw_sentence=s.split('\t')[0]
    sentence=[x for x in list(''.join(set(''.join(raw_sentence)))) if x not in insignificant_tokens]
    model_line =str(idx+1)
    #add unigrams
    for ch in sentence:
        feature_list.append(character_numbers[ch])
        
    #add bigrams
    bigrams=set()
    for words in raw_sentence.split(' '):
        bigrams.update([x.lower() for x in[words[i:i+2] for i in range(len(words)-1)]])
    bigrams_list=list([x for x in bigrams if (x[0] not in insignificant_tokens)and(x[1] not in insignificant_tokens)])
    for bg in bigrams_list:
        feature_list.append(character_numbers[bg])
    
    feature_list=list(set(feature_list))
    feature_list.sort()
    for ch in feature_list:
        model_line+=" "+ str(ch)+ ":" + str(1)
    return model_line+"\n"

#these tokens will be skipped for both creating unigrams and bigrams  
insignificant_tokens=[' ','!', '"','#','$','%','&','*','+','-','(',')',',','.','/','0','1','2','3','4','5','6','7','8',
                      '9',';','<','>','=','?','@','|','«','»','`','[',']',"'",'\\']
language_ids=['bg','bs','cz','es-AR','es-ES','hr','id','mk','my','pt-BR','pt-PT','sk','sr']
with open ("Corpus/Raw Corpus.txt") as f:
    corpus = f.readlines()
languages=[]
corpus_s=list(''.join(set(''.join(corpus))))
corpus_s=[x for x in corpus_s if x not in insignificant_tokens]
character_numbers={}
i=0
for s in corpus_s:
    i+=1
    character_numbers[s]=i 

bigrams=set() #create bigrams as set to for uniqueness
for s in corpus:
    for words in s.split(' '):
        bigrams.update([x.lower() for x in[words[i:i+2] for i in range(len(words)-1)]]) #add each bigrams in the sentences
#skip bigrams contains skipped tokens 
bigrams_list=list([x for x in bigrams if (x[0] not in insignificant_tokens)and(x[1] not in insignificant_tokens)])
for bg in bigrams_list:
    i+=1
    character_numbers[bg]=i
for i in range(13): #divides data set as languages
    languages.append("")
    start=i*2000
    end=(i+1)*2000
    languages[i]=corpus[start:end] #2k sentence for each language
training_set=[]
test_set=[]
training_model=[]
test_model=[]
for idx,l in enumerate(languages):
    random.shuffle(l)
    
    training_partition=l[0:1800]
    training_set.extend(training_partition)
    
    for s in training_partition:
        training_model.append(generate_model_line(s,idx))
    
    test_partition=l[1800:2000]
    test_set.extend(test_partition)
    
    for s in test_partition:
        test_model.append(generate_model_line(s,idx))


# In[2]:


def write_files(file_name,array):
    with open (file_name, mode='wt') as t_file:
        for item in array:
            t_file.write(item)
print(len(training_set))
print(len(test_set))
write_files("SVM/IncreasedFeatures/TrainingModel-SVM.txt",training_model)
write_files("SVM/IncreasedFeatures/TrainingSet-SVM.txt",training_set)
write_files("SVM/IncreasedFeatures/TestSet-SVM.txt",test_set)
write_files("SVM/IncreasedFeatures/TestModel-SVM.txt",test_model)


# In[3]:


p=subprocess.Popen(['SVM/svm_multiclass_learn', '-c', '5000' ,'SVM/IncreasedFeatures/TrainingModel-SVM.txt' ,'SVM/IncreasedFeatures/Model'],
                 stdout=subprocess.PIPE)

p.wait()



# In[4]:


p=subprocess.Popen(['SVM/svm_multiclass_classify', 'SVM/IncreasedFeatures/TestModel-SVM.txt' ,'SVM/IncreasedFeatures/Model','SVM/IncreasedFeatures/predictions.txt'],
                 stdout=subprocess.PIPE)
for line in p.stdout:
    print(line)
p.wait()


# In[5]:


with open ("SVM/IncreasedFeatures/predictions.txt") as f:
    predictions = f.readlines()
predictions=[i.split(' ')[0] for i in predictions]
metrics={}
for idx,lang_id in enumerate(language_ids):
    false_negatives=len([x for x in predictions[idx*200:(idx+1)*200] if int(x) != (idx+1)])
    true_positives=200-false_negatives
    if idx==0: #count false positives on proceeding predictions
        false_positives=len([x for x in predictions[200:len(predictions)] if int(x) ==1])
    elif idx==12:  #count false positives on preceeding predictions
        false_positives=len([x for x in predictions[0:2400] if int(x)==13])
    else: #count false positives on both preceeding and proceeding predictions
        false_positives=[x for x in predictions[(idx-1)*200:idx*200] if int(x) == (idx+1)]
        false_positives=len(false_positives + [x for x in predictions[(idx+1)*200:2600] if int(x) == (idx+1)])
    true_negatives=2400-false_positives
    metrics[lang_id]={"tp":true_positives,"fn":false_negatives,"tn":true_negatives,"fp":false_positives}
for key,value in metrics.iteritems():
    print(key)
    print("True positives: " + str(value["tp"]))
    print("False positives: " + str(value["fp"]))
    print("True negatives: " + str(value["tn"]))
    print("False negatives: " + str(value["fn"]))
    


# In[6]:


total_tp=0.0
total_fp=0.0
total_fn=0.0
total_tn=0.0
total_precision=0.0
total_recall=0.0
total_f1score=0.0
for key in metrics.keys():
    tp=metrics[key]["tp"]
    fp=metrics[key]["fp"]
    fn=metrics[key]["fn"]
    tn=metrics[key]["tn"]
    precision=tp/float(tp+fp)
    recall=tp/float(tp+fn)
    f1_score=(2*recall*precision)/float(recall+precision)
    total_precision+=precision
    total_recall+=recall
    total_f1score+=f1_score
    
    total_tp+=tp
    total_fp+=fp
    total_fn+=fn
    total_tn+=tn
    
mic_prec=total_tp/float(total_tp+total_fp)
mic_recall=total_tp/float(total_tp+total_fn)
print("Micro-averaged precision: " + str(mic_prec))
print("Micro-averaged recall: " + str(mic_recall))
print("Micro-averaged f1-score: " + str((2*mic_prec*mic_recall)/float(mic_prec+mic_recall)))
print("")
print("Macro-averaged precision: " + str(total_precision/13.0))
print("Macro-averaged recall: " + str(total_recall/13.0))
print("Macro-averaged f1-score: " + str(total_f1score/13.0))
print("")
print("Total accuracy: "+ str(total_tp/2600.0))
print("Accuracies for languages:")
for key,value in metrics.iteritems():
    print(key + str(": ")+str(value["tp"]/200.0))
print("")
print("fp: "+str(total_fp))
print("tp: "+str(total_tp))
print("fn: "+str(total_fn))
print("tn: "+str(total_tn))

metrics


# In[7]:


for key,value in metrics.iteritems():
    print(key + "\t"+ str(value["tp"]) + "\t"+ str(value["fp"])+"\t"+str(value["tn"])+"\t"+str(value["fn"]))
    


# In[8]:


print("Accuracies for languages:")
for key,value in metrics.iteritems():
    print(str(value["tp"]/200.0))
print("")

