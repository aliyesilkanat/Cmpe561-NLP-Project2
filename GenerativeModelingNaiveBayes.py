
# coding: utf-8

# In[8]:


import random
#these tokens will be skipped while calculating probability
insignificant_tokens=[' ','!', '"','#','$','%','&','*','+','-','(',')',',','.','/','0','1','2','3','4','5','6','7','8',
                      '9',';','<','>','=','?','@','|','«','»','`','[',']',"'",'\\']
language_ids=['bg','bs','cz','es-AR','es-ES','hr','id','mk','my','pt-BR','pt-PT','sk','sr']

class Language: #for each language in data set, we create a Language class
   
    def __init__ (self,lang_id):
        self.chars={} # storing characters and their counts
        self.prob={} #storing p(c|l) probability for each character
        # corresponding language id 
        self.lang_id=lang_id
    def calculate_probability(self,sentence):
        s=list(sentence)
        s=[x for x in s if x not in insignificant_tokens]
        total=0
        for ch in s:
            if self.chars.has_key(ch):
                total+=self.prob[ch]
            else: 
                return 0
        return total
    def get_chars(self):
        return self.chars
    def get_prob(self):
        return self.prob
    def get_lang_id(self):
        return self.lang_id
with open ("Corpus/Raw Corpus.txt") as f:
    corpus = f.readlines()


languages=[]
model=[]
for i in range(13): #divides data set as languages
    languages.append("")
    start=i*2000
    end=(i+1)*2000
    languages[i]=corpus[start:end] #2k sentence for each language

training_set=[]
test_set=[]
for idx,l in enumerate(languages):
    lang=Language(language_ids[idx])
    random.shuffle(l)     
    
    training_partition=l[0:1800]
    training_set.extend(training_partition)
 
    training_partition=[i.split('\t')[0] for i in training_partition] #remove language identifier at the last of the sentences
    for sentence in training_partition:
        for letter in sentence:
            if letter not in insignificant_tokens:
                if lang.get_chars().has_key(letter):
                    nominator=lang.get_chars()[letter]
                else:
                    nominator=0 #laplace 
                nominator+=1
                lang.get_chars()[letter]=nominator
                
    test_partition=l[1800:2000]
    test_set.extend(test_partition)
    test_part=[i.split('\t')[0] for i in test_partition] #remove language identifier at the last of the sentences
    
    unk=list(''.join(set(''.join(test_part))))
    unk=[x for x in unk if x not in insignificant_tokens]
    unk=[x for x in unk if x not in lang.get_chars().keys()]
    if len(unk)!=0: # add unknowns characters in test set for smoothing
        for l in unk:
            lang.get_chars()[l]=0
        
    for ch in lang.chars: #laplace smoothing
        lang.get_chars()[ch]+=1
    
    denominator=sum(lang.get_chars().values())+len(lang.get_chars().values()) #for performance
    for letter in lang.get_chars(): #calculate probabilities
        lang.get_prob()[letter]=(lang.get_chars()[letter]+1)/float(denominator)
  

    model.append(lang)


# In[9]:


i=0
predictions=[]
expected=[]
for sentence in test_set:
    s=sentence.split('\t')
    probabilities={}
    expected.append(language_ids.index(s[1].strip())+1)
    for l_model in model: #foreach language calculate the probability of the given sentence
        probabilities[l_model.get_lang_id()]=l_model.calculate_probability(s[0])
    #find one language having most likely given sentence     
    predictions.append(language_ids.index(max(probabilities.items(), key=lambda k: k[1])[0])+1) 


# In[10]:


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


# In[11]:


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


# In[ ]:



def write_files(file_name,array):
    with open (file_name, mode='wt') as t_file:
        for item in array:
            t_file.write(item)
print(len(training_set))
print(len(test_set))
write_files("Training set.txt",training_set)
write_files("Test set.txt",test_set)

