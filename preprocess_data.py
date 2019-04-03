from nltk.corpus import brown
import nltk
import pickle
import time
import json
import os

corpus=[]
for genre in brown.categories():
    corpus += brown.tagged_sents(categories=genre)       

tag = set()
for sen in corpus:
    for wordset in sen:
        tag.add(wordset[1])

f=open("brown_taglist.txt","w")
for each in tag:
    f.write(each+'\n')
f.close()

## create word_seg.txt and label_seg.txt
temp1 = open('C:\Users\a\Desktop\data\word_seg.txt','w',encoding='utf-8')
temp2 = open('C:\Users\a\Desktop\data\label_seg.txt','w',encoding='utf-8')

for sen in corpus:
    for wordset in sen:
        temp1.write(wordset[0]+' ')
        temp2.write(wordset[1]+' ')
    temp1.write('\n')
    temp2.write('\n')
temp1.close()
temp2.close()


## NLTK n-gram tagger
train_sents = corpus[:50000]
test_sents = corpus[50000:]
start=time.time()
t0 = nltk.DefaultTagger('NN')  
t1 = nltk.UnigramTagger(train_sents,backoff=t0)  
t2 = nltk.BigramTagger(train_sents,backoff=t1)  
#print('run time:', time.time()-start)
#print('training accuracy', t2.evaluate(train_sents))
#print('testing accuracy', t2.evaluate(test_sents))

text = nltk.word_tokenize("And now for something completely different...")
t2.tag(text)
nltk.pos_tag(text)

with open('C:\Users\a\nltk_data\help\tagsets\brown_tagset.pickle','rb') as f:
    ls = pickle.load(f)
ls.keys()

f=open('brown_tagsets.txt','w',encoding='utf-8')
for key, value in ls.items():
    f.write(key+' // '+value[0]+ '. E.g., ' + value[1] + '\n')
f.close()

# see those who are in the corpus but not in brown_tagsets.txt
unknowntag = set()
for sen in corpus:
    for wordset in sen:
        if wordset[1] not in ls:
            unknowntag.add(wordset[1])
#print(unknowntag)

os.chdir('../data')

#split corpus into training and testing data 
with open('data_seg.txt','r',encoding='utf-8') as f:
    data_seg = f.readlines()
n=50000
data_seg_train = data_seg[:n]
data_seg_test = data_seg[n:]

with open('data_seg_train.txt','w',encoding='utf-8') as f:
    for sen in data_seg_train:
        f.write(sen)
        
with open('data_seg_test.txt','w',encoding='utf-8') as f:
   for sen in data_seg_test:
        f.write(sen)

with open('lebel_seg.txt','r',encoding='utf-8') as f:
    label_seg = f.readlines()
label_seg_train = label_seg[:n]
label_seg_test = label_seg[n:]

with open('label_seg_train.txt','w',encoding='utf-8') as f:
    for sen in label_seg_train:
        f.write(sen)
        
with open('label_seg_test.txt','w',encoding='utf-8') as f:
    for sen in label_seg_test:
        f.write(sen)

#create label directory
with open('label_seg_train.txt','r',encoding='utf-8') as f:
    data = f.readlines()
with open('label_seg_test.txt','r',encoding='utf-8') as f:
    data2 = f.readlines()
posls=set()
for sen in data:
    sen = sen.strip(' \n')
    wordls = sen.split(' ')
    for word in wordls:
        posls.add(word)
for sen in data2:
    sen = sen.strip(' \n')
    wordls = sen.split(' ')
    for word in wordls:
        posls.add(word)
print(len(posls),posls)

posdt={}
count=0
for tag in posls:
    posdt[tag]=count
    count+=1
posdt.values()

tag_dt={}
for key in posdt.keys():
    vec = [0]*len(posls)
    vec[posdt[key]] = 1
    tag_dt[key]=vec
    
with open('tagdt.json', 'w') as f:
    json.dump(tag_dt,f)
    
with open('posdt.json', 'w') as f:
    json.dump(posdt,f)
