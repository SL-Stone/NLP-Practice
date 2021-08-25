'''
统计n篇.txt文档的中文分词的总tfidf和词频
'''

from collections import defaultdict
import re

words_freq = defaultdict() #在所有文档出现的词频
words_docs_count = defaultdict() #出现该词的所有文档数
doc_words_freq = defaultdict()  #记录每个文档出现的所有词及词频
doc_words_tfidf = defaultdict()  #记录每个文档出现的所有词及tfidf
all_words_tfidf = defaultdict()  #记录所有文档出现的所有词及tfidf总和

filelist = ["D:\\test\\1.txt","D:\\test\\2.txt","D:\\test\\3.txt"]  #所有源文档列表
file_freq = "D:\\test\\freq.txt"
file_tfidf = "D:\\test\\tfidf.txt"

num_files = len(filelist)

for file in filelist:
    doc_words_freq[file]=defaultdict()
    doc_words_tfidf[file]=defaultdict()
    
for file in filelist:
    words_freq_doc = []
    with open(file, 'r', encoding = 'utf8') as f: 
        flag = defaultdict() #记录有没有被统进文档数里
        for line in f.readlines():
            segments = re.sub(r'[A-Za-z0-9]+','',line) #去数字
            seg_list = segments.strip().split(" ")
            for word in seg_list:
                if word.strip():
                    print(word)
                    if word not in doc_words_freq[file]: #统计每篇文档自己的词频
                        doc_words_freq[file][word] = 1
                    else:
                        doc_words_freq[file][word] += 1
                        
                    if word not in flag:  #统计文档数
                        flag[word] = 1                   
                        if word not in words_docs_count:
                            words_docs_count[word] = 1                   
                        else:
                            words_docs_count[word] += 1
                            
                    if word not in words_freq: #统计所有文档的词频
                        words_freq[word] = 1
                    else:
                        words_freq[word] += 1
    print(doc_words_freq[file])
    
order_words_freq = sorted(words_freq.items(),key=lambda x:x[1],reverse=True)  
with open(file_freq, 'a+', encoding = 'utf8') as f_freq: #写词频文件
    for k,v in order_words_freq:
        f_freq.write("%s %d\n" % (k,v))

for file in filelist:  #统计每个文档的tfidf
    max_freq_doc = doc_words_freq[file][max(doc_words_freq[file],key = doc_words_freq[file].get)]
    for k,v in doc_words_freq[file].items():
        doc_words_tfidf[file][k] = (v*num_files)/(max_freq_doc*words_docs_count[k])
        if k not in all_words_tfidf:
            all_words_tfidf[k] = doc_words_tfidf[file][k]
        else:
            all_words_tfidf[k] += doc_words_tfidf[file][k]
order_words_tfidf = sorted(all_words_tfidf.items(),key=lambda x:x[1],reverse=True)  
with open(file_tfidf, 'a+', encoding = 'utf8') as f_tfidf: #写tfidf文件
    for k,v in order_words_tfidf:
        f_tfidf.write("%s %.5f\n" % (k,v))
