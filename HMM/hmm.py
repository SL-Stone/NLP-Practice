from collections import defaultdict
import pdb
import time
import numpy as np
import math

file_path = "PeopleDaily1998_partial.txt"

word2id = {'UNK': 0,} # hash management
pos2id = {'POS_UNK': 0,}
sentences = [] # Chinese word, not word_id
pos_list = [] # English letters, not pos_id
start_freq = defaultdict()
words_num = 0

def read_txt(file_path):
    global word2id
    global pos2id
    global start_freq
    global words_num
    global sentences
    global pos_list
    word_count = 1
    pos_count = 1
    with open(file_path, 'r', encoding='gbk') as f: #gb2312编码解码不能用gb2312编码 'gb2312' codec can't decode byte 0xe9
        end = False
        line = f.readline()
        while not end:
            if(line.strip() == ''): # strip()过滤[空白]
                line = f.readline()
                if(line.strip() == ''): # 连续两个空行则认为到达文件尾
                    end = True
            else:
                sentences.append([])
                pos_list.append([])
                words = line.split()[1:] #将第一个日期过滤掉
                pos_start = words[0].split('/')[1]
                if pos_start in start_freq:
                    start_freq[pos_start] += 1
                else:
                    start_freq[pos_start] = 1

                for ww in words: # split按空白分割，包含空格、\n、tab
                    words_num += 1
                    pair = ww.split('/')
                    sentences[-1].append(pair[0])
                    pos_list[-1].append(pair[1])
                    if pair[0] not in word2id:
                        word2id[pair[0]] = word_count
                        word_count += 1
                    if pair[1] not in pos2id:
                        pos2id[pair[1]] = pos_count
                        pos_count += 1

            line = f.readline()

class HMMmodel(object):
    def __init__(self, word2id, pos2id):
        self.word2id = word2id
        self.pos2id = pos2id
        self.word_size = len(word2id)
        self.pos_size = len(pos2id)

        self.id2pos = {v:k for k,v in self.pos2id.items()} # dict( (v,k) for k,v in self.pos2id.items() )

        # initial 初始状态矩阵，第一个词性
        self.initial = np.zeros(self.pos_size)
        # transition 状态转移矩阵词性到词性的转移
        self.transition = np.zeros((self.pos_size, self.pos_size))
        # emission 发射矩阵词性到词
        self.emission = np.zeros((self.pos_size, self.word_size))

    def calculate_matrix(self, sentences, pos_list, smooth = False): # initialize 3 matrices
        sentences_num = len(sentences)

        for k, v in start_freq.items():
            pos_id = pos2id[k]
            if smooth is False:
                self.initial[pos_id] = v / sentences_num
            else: # Laplace Smoothing
                self.initial[pos_id] = (v + 1) / (sentences_num + self.pos_size)
        
        for sentence, pos in zip(sentences, pos_list):
                pos_pre_id = None
                for w, p in zip(sentence, pos):
                    word_id = word2id[w]
                    pos_id = pos2id[p]
                    if pos_pre_id is not None:
                        self.transition[pos_pre_id][pos_id] += 1
                    pos_pre_id = pos_id
                    self.emission[pos_id][word_id] += 1

        # Transfer frequency to relative frequency / probability for every item in 3 matrices. 矩阵中的频数转概率
        # UNK specially
        self.transition[0] = 1 / self.pos_size
        self.emission[0] = 1 / self.word_size
        for i in range(1, self.pos_size):
            pos_freq = np.sum(self.transition[i])
            word_freq = np.sum(self.emission[i])
            if not smooth:
                self.transition[i] = self.transition[i] / pos_freq
                self.emission[i] = self.emission[i] / word_freq
            else:
                self.transition[i] = (self.transition[i] + 1) / (pos_freq + self.pos_size)
                self.emission[i] = (self.emission[i] + 1) / (word_freq + self.word_size)

    def _forward(self, obs_id_seq, task="sum"):
        word_len = len(obs_id_seq)
        forward = np.zeros((word_len, self.pos_size))

        # Special calculation for staring boundary forward[0] 
        for i in range(self.pos_size):
            forward[0][i] = self.initial[i]*self.emission[i][obs_id_seq[0]]
        
        for index in range(1, word_len):
            for i in range(self.pos_size):
                sum_prob = 0
                for j in range(self.pos_size):
                    sum_prob += forward[index-1][j] * self.transition[j][i]
                forward[index][i] = sum_prob * self.emission[i][obs_id_seq[index]]
                # forward[index, i] = np.dot(forward[index-1,:], self.transition[:, i]) * self.emission[i][word_id_list[index]]
        return forward

    def _backward(self, obs_id_seq, task="sum"):
        word_len = len(obs_id_seq)
        backward = np.zeros((word_len, self.pos_size))

        # Special calculation for staring boundary backward[-1] 
        for i in range(self.pos_size):
            backward[-1,i] = 1
        for index in range(word_len-2, -1, -1): # the last '-1' represents reverse/descending order -> reversed(range(word_len-1))
            for i in range(self.pos_size):
                sum_prob = 0
                for j in range(self.pos_size):
                    sum_prob += self.transition[i][j] * backward[index+1][j] * self.emission[j][obs_id_seq[index+1]]
                backward[index][i] = sum_prob
            # note that backward[0] does not calculate emission[i][0th word]
            # note that we don't need backward[0]. for index in range(word_len-2, 0, -1)
        
        return backward
        

    def expectation_max(self, sentences, pos_list, threshold=1e-15, epoch=None):
        word_id_lists = []
        for sentence in sentences:
            word_len = len(sentence)
            word_id_list = []
            for index in range(0, word_len):
                word_id = 0 # 'UNK'
                if(sentence[index] in self.word2id):
                    word_id = self.word2id[sentence[index]]
                word_id_list.append(word_id)
            word_id_lists.append(word_id_list)

        prob = 0
        iter = 0
        if epoch is None:
            while prob < threshold:
                prob = self.baum_welch(word_id_lists, pos_list)
                print("The probability of {} epoch: {}".format(iter+1, prob))
                iter+=1
        else:
            while iter < epoch:
                prob = self.baum_welch(word_id_lists, pos_list)
                print("The probability of {} epoch: {}".format(iter+1, prob))
                iter+=1

        print("Expectation-Maximization process finished.")
        print("The current probability is", prob)
    
    def baum_welch(self, word_id_lists, pos_list, keep_old = 0.95): 
        # keep_old 为了防止单条句子上对矩阵更新，造成下一条句子的hmm生成概率为0，导致矩阵无法更新nan
        # 在更新矩阵时，采用keep_old的权重保留上一步的矩阵单数
        prob_cur = 0
        start = time.time()
        www=-1 # record the current sentence id.
        for word_id_list, pos in zip(word_id_lists, pos_list):
            www+=1
            word_len = len(word_id_list)
            forward = self._forward(word_id_list)
            backward = self._backward(word_id_list)
            
            obs_prob = np.sum(forward[-1])
            if obs_prob <= 0:
                print("P(O|lambda) = 0. Cannot optimize.")
                print("Maybe caused by too long sequence. OR overlapping on previous partial data distribution.")
                print("Sequence Length: ", word_len)
                print(www)
                continue
            prob_cur += obs_prob

            # E PROCEDURE
            kexi = np.zeros((word_len-1, self.pos_size, self.pos_size)) # t时刻，隐状态i转移到j的概率
            for index in range(word_len - 1):
                sum_prob = 0
                for i in range(self.pos_size):
                    for j in range(self.pos_size):
                        kexi[index][i][j] = forward[index][i] * backward[index+1][j] * self.emission[j][word_id_list[index+1]]
                        sum_prob += kexi[index][i][j]
                kexi[index] /= sum_prob
            
            gamma_t = np.zeros((word_len, self.pos_size))
            gamma_t[:-1] = np.sum(kexi, -1) # t时刻，是隐状态i的概率 [word_len, pos_size], sum(gamma[i])=1
            gamma_t[-1] = forward[-1] / np.sum(forward[-1])

            # M PROCEDURE

            #self.initial = gamma_t[0].copy()
            #self.initial = 0.9*self.initial + 0.1*gamma_t[0]
            # Do not update self.initial


            gamma_sum_A = np.sum(gamma_t[:-1,:], 0).reshape(-1,1) # pos id有没有在训练集中出现过，出现则>0，没有出现过则=0
            # (pos_size, 1)
            rows_to_keep_Apos = (gamma_sum_A == 0)
            selected_transition = self.transition * (gamma_sum_A > 0)
            selected_transition_denominator = np.sum(selected_transition,1).reshape(-1,1)
            selected_transition_denominator[gamma_sum_A == 0] = 1
            gamma_sum_A[gamma_sum_A == 0] = 1 # 0/1仍为0
            # next_transition = np.sum(kexi, axis=0) / gamma_sum_A # [pos_size, pos_size] / [pos_size, 1] 广播
            
            # next_transition = (selected_transition + np.sum(kexi, axis=0)) / (np.sum(selected_transition,1).reshape(-1,1) + gamma_sum_A)
            next_transition = keep_old*selected_transition + (1-keep_old)*np.sum(kexi, axis=0) / gamma_sum_A
            # 当前没有用到pos1, 任何时刻的词对应的pos1概率为0，则pos1对应的行保持原来的数值不变
            self.transition = self.transition * rows_to_keep_Apos + next_transition

            gamma_sum_B = np.sum(gamma_t, 0)[:, np.newaxis]
            rows_to_keep_Bpos = (gamma_sum_B == 0)
            selected_emissition = self.emission * (gamma_sum_B > 0)
            gamma_sum_B[gamma_sum_B == 0] = 1

            observe = np.zeros((word_len, self.word_size))
            observe[range(word_len), word_id_list] = 1
            # next_emission = np.dot(gamma_t.T, observe) / gamma_sum_B
            
            # next_emission = (selected_emissition + np.dot(gamma_t.T, observe)) / (np.sum(selected_emissition,1).reshape(-1,1) + gamma_sum_B)
            next_emission = keep_old*selected_emissition + (1-keep_old)*np.dot(gamma_t.T, observe) / gamma_sum_B
            self.emission = self.emission * rows_to_keep_Bpos + next_emission


        end = time.time()
        print(end-start)
        return prob_cur / len(word_id_lists)


    def viterbi(self, sentence_input): # Viterbi/Dynamic Programming, using forward algorithm
        
        pos_path0 = {key: [] for key in range(self.pos_size)}
        pos_path1 = {key: [] for key in range(self.pos_size)}
        dp0 = np.zeros(self.pos_size) # todo: 将词性标注序列的概率积转换为-log
        dp1 = np.zeros(self.pos_size)

        # Initial Status
        start_word_id = 0 # 'UNK'
        if(sentence_input[0] in self.word2id):
            start_word_id = self.word2id[sentence_input[0]]
        
        for i in range(self.pos_size):
            if self.initial[i] != 0 and self.emission[i][start_word_id] != 0:
                dp0[i] = self.initial[i]*self.emission[i][start_word_id]
                pos_path0[i].append(i)
            
        for index, word in enumerate(sentence_input[1:]):
            word_id = 0 # 'UNK'
            if(word in self.word2id):
                word_id = self.word2id[word]
            for i in range(self.pos_size):
                dp1[i] = 0
                if self.emission[i][word_id] == 0:
                    continue
                emit_posb = self.emission[i][word_id]
                max_pos_id = -1
                max_trans_posb = 0
                for j in range(self.pos_size):
                    if dp0[j] != 0:
                        if self.transition[j][i] == 0:
                            continue
                        trans_posb = dp0[j] * self.transition[j][i]
                        if(trans_posb > max_trans_posb):
                            max_pos_id = j
                            max_trans_posb = trans_posb
                if max_trans_posb != 0:
                    dp1[i] = max_trans_posb * emit_posb
                    assert len(pos_path0[max_pos_id]) == index + 1
                    pos_path1[i] = pos_path0[max_pos_id].copy()
                    pos_path1[i].append(i)
                
                #else: 用i做隐状态，没有路

            dp0 = dp1.copy()
            pos_path0 = pos_path1.copy()
            pos_path1 = {key: [] for key in range(self.pos_size)}

        last_pos_id = np.argmax(dp0)
        return np.max(dp0), pos_path0[last_pos_id]

    def transfer_id2pos(self, pos_id_list):
        return [self.id2pos[i] for i in pos_id_list]

    def accuracy(self, pos_predicted, pos_true): # exact matching accuracy
        total_pos = 0
        match_pos = 0
        for preds, tags in zip(pos_predicted, pos_true):
            total_pos += len(preds)
            for pred, tag in zip(preds, tags):
                if pred == tag :
                    match_pos += 1
                else:
                    a = 0
        return match_pos / total_pos


if __name__ == "__main__":

    read_txt(file_path)
    
    hmm = HMMmodel(word2id, pos2id)

    # initialize 3 matrices
    hmm.calculate_matrix(sentences, pos_list)

    
    # Decoding
    #input = ["联合政府", "撤军", "仍", "无", "具体", "方案"] #以/j  撤军/v  仍/d  无/v  具体/a  方案/n
    input = ["北京", "举行", "新年", "音乐会"] #北京/ns  举行/v  新年/t  音乐会/n  

    pos_pred = []
    for sentence in sentences[:100]:
        _, pos_p = hmm.viterbi(sentence)
        pos_pred.append(hmm.transfer_id2pos(pos_p))
    print("accuracy without training parameters:", hmm.accuracy(pos_pred, pos_list[:100]))

    # Estimation
    # hmm.expectation_max([sentences[1]], [pos_list[1]], threshold=1e-12)
    hmm.expectation_max(sentences[:20], pos_list[:20])
    
    pos_pred = []
    for sentence in sentences[:100]:
        _, pos_p = hmm.viterbi(sentence)
        pos_pred.append(hmm.transfer_id2pos(pos_p))
    print("accuracy after training parameters:", hmm.accuracy(pos_pred, pos_list[:100]))
  
