#!/usr/bin/env python
# coding: utf-8

# # 一 、Rule Based:Syntax tree

# ## Example 1

# In[1]:


import random


# In[2]:


two_number = """
numbers => num numbers | num
num => 0 | 1 | 2 | 3 | 4
"""

def num():
    return random.choice("0 | 1 | 2 | 3 | 4".split('|'))

def two_num():
    return num() + num()

def numbers():
    if random.random() < 0.5:     # 产生0到1之间随机浮点数
        return num()
    else:
        return num() + numbers()


# In[3]:


num()


# In[4]:


two_num()


# In[5]:


for i in range(10):
    print(numbers())


# ## Example 2

# In[6]:


sentence = """
sentence => noun_phrase verb_phrase
noun_phrase => Article Adj* noun
Adj* => Adj | Adj Adj*
verb_phrase => verb noun_phrase
Article => 一个 | 这个
noun => 女人 |  篮球 | 桌子 | 小猫
verb => 看着  | 坐在 | 听着 | 看见
Adj =>  蓝色的 | 好看的 | 小小的
"""


# In[7]:


import random


# In[8]:


def verb():
    return random.choice("看着 | 坐在 | 听着 | 看见".split('|'))

def noun():
    return random.choice("女人 |  篮球 | 桌子 | 小猫".split('|'))


def verb_phrase():
    return verb() + noun()


# In[9]:


verb_phrase()


# In[10]:


def adj():  return random.choice('蓝色的 | 好看的 | 小小的'.split('|')).split()[0]
def adj_star():
    return random.choice([lambda : '', lambda : adj() + adj_star()])()


# In[11]:


adj_star()


# ## Example 3 Definition grammar

# In[12]:


adj_grammar = """
Adj* = Adj | Adj Adj*
Adj = 蓝色的 | 好看的 | 小小的
"""


# In[13]:


def generation_by_grammar(grammar_str:str, target, split = '='):
    rules = dict()  # 用于创建一个字典 key is @statement, value is @expression
    
    for line in grammar_str.split('\n'):
        if not line: continue   # skip the empty line
        
        stmt, expr = line.split(split)
        rules[stmt.strip()] = expr.split('|')
        
        
    generated = generate(rules, target=target)
    return generated


# In[15]:


def generate(grammar_rule, target):
    if target in grammar_rule:
        candidates = grammar_rule[target]
        candidate = random.choice(candidates)
        return ' '.join(generate(grammar_rule, target=c.strip()) for c in candidate.split())
    else:
        return target


# In[17]:


generation_by_grammar(adj_grammar,target='Adj*')


# In[18]:


generation_by_grammar(adj_grammar, target='Adj')


# In[19]:


host = """
host = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号 ,
数字 = 单个数字 | 数字 单个数字

单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
玩玩 = null
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？
"""


# In[20]:


generation_by_grammar(host, target='host')


# ## 作业

# In[72]:


sentence = """
sentence = sub day jud jie 结尾
sub = sub1 sub | sub1
sub1 = 哈瑞， | Jone， | 小王， | 小明，
day = 今天 | 昨天 | 后天
jud = 是 | 不是
jie = 情人节 | 星期一 | 国庆
结尾 = 吗？ | 嘛？
"""


# In[22]:


generation_by_grammar(sentence, target='sentence')


# # 二、Probability Based:Language Model

# $$ language\_model(String) = Probability(String) \in (0, 1) $$

# $$ Pro(w_1 w_2 w_3 w_4) = Pr(w_1 | w_2 w_3 w_ 4) * P(w2 | w_3 w_4) * Pr(w_3 | w_4) * Pr(w_4)$$ 

# $$ Pro(w_1 w_2 w_3 w_4) \sim Pr(w_1 | w_2 ) * P(w2 | w_3 ) * Pr(w_3 | w_4) * Pr(w_4)$$ 

# ### Pr（其实就和随机森林原理一样）

# ->Pr（其实&就和&随机森林&原理&一样）

# -> Pr（其实|就和&随机森林&原理&一样）Pr(就和&随机森林&原理&一样）

# -> Pr（其实|就和)Pr(就和&随机森林&原理&一样） 1-gram

# -> Pr（其实|就和)Pr(就和|随机森林&原理&一样）Pr(随机森林&原理&一样）

# -> Pr（其实|就和)Pr(就和|随机森林）Pr(随机森林&原理&一样）

# -> Pr（其实|就和)Pr(就和|随机森林）Pr(随机森林|原理&一样）Pr（原理&一样）

# -> Pr（其实|就和)Pr(就和|随机森林）Pr(随机森林|原理）Pr（原理&一样）

# -> Pr（其实|就和)Pr(就和|随机森林）Pr(随机森林|原理）Pr（原理|一样）Pr（一样）

# $$Pr(sentence) = Pr(w_1w_2w_3w_4) = \prod_i^{n} \frac{\# w_iw_{i+1}}{\# w_{i+1}} * Pr(w_n) $$

# how to get $ Pr(w1 | w2 w3 w4) $ ?

# In[23]:


import pandas as pd


# In[26]:


filename = 'E:\\开课吧npl\\NLP\\NLP_Class1\\sqlResult_1558435.csv'  # 在windows中， \ 容易理解为转义符 所以都改为 \\


# In[28]:


content = pd.read_csv(filename, encoding='gb18030')


# >python读写文件的几种方式
# 
# >1.read_csv 从文件，url，文件型对象中加载带分隔符的数据。默认分隔符为逗号
# 
# >2.read_table 从文件，url，文件型对象中加载带分隔符的数据。默认分隔符为制表符（“\t”）
# 
# >3.read_fwf 读取定宽列格式数据（也就是没有分隔符）
# 
# >4.read_cliboard 读取剪切板中的数据，可以看做read_table的剪切板。在将网页转换为表格时很有用
# 
# >[python pandas中文件的读写](https://blog.csdn.net/qq_24084925/article/details/79608684)

# In[29]:


content.head()       # head()只读取前5行数据


# In[30]:


articles = content['content'].tolist()   


# >tolist()函数将数组或者矩阵转换成列表 [tolist()用法](https://www.cnblogs.com/Aaron12/p/9042687.html)

# In[31]:


len(articles)


# In[32]:


import re    # re是正则化模块的包


# In[33]:


import jieba    #jieba是用来分词的包


# >[re模块](https://www.cnblogs.com/zjltt/p/6955965.html)    [jieba模块](https://blog.csdn.net/codejas/article/details/80356544)

# In[79]:


list(jieba.lcut('这个是用来做汉语分词的')


# In[35]:


def token(string):                      # 作用是去除乱七八糟符号
    # we will learn the regular expression next course.
    return re.findall('\w+', string)          


# In[36]:


articles[0]


# In[37]:


token(articles[0])


# In[38]:


from collections import Counter  # counter() 就是collections中的一个计数器


# >[collections](https://www.liaoxuefeng.com/wiki/897692888725344/973805065315456)

# In[39]:


with_jieba_cut = Counter(jieba.cut(articles[110]))


# In[40]:


print(with_jieba_cut)


# In[41]:


with_jieba_cut.most_common()[:10]    # .most_common()函数在collections模块中，功能是从高到底排序


# In[42]:


''.join(token(articles[110]))   # .join()函数把articles[100]中的 ， 都去掉了


# In[43]:


articles_clean = [''.join(token(str(a)))for a in articles]    # .join()函数把整个articles的中 ， 都去掉了 形成一个列表（可以取ariticle[：10]输出试一下|）


# In[44]:


len(articles_clean)


# In[45]:


with open('article_9k.txt', 'w') as f:
    for a in articles_clean:
        f.write(a + '\n')


# > 在AI的问题里面，大部分时间都是在数据的预处理，我们要养成一个习惯：把重要信息及时的保存到硬盘里
# 
# >[with_open() as f 读写操作](https://blog.csdn.net/wzhrsh/article/details/101629075)

# In[46]:


import jieba


# In[47]:


def cut(string): 
    return (jieba.cut(string))


# In[48]:


ALL_TOKEN = cut(open('article_9k.txt').read())


# In[49]:


TOKEN = []  # ALL_TOKEN被cut以后形成列表


# In[50]:


for i, t in enumerate(ALL_TOKEN):
    if i > 1000000: break
    if i % 1000 == 0: print(i)
    TOKEN.append(t)


# >[enumerate()函数的用法](https://www.runoob.com/python/python-func-enumerate.html)

# In[81]:


len(TOKEN) # 单词数


# In[82]:


with open('article_9k_cut_txt', 'w') as f:   # 这个好像是在保存切词后的文件
    pass


# $$Pr(sentence) = Pr(w_1w_2w_3w_4) = \prod_i^{n} \frac{\# w_iw_{i+1}}{\# w_{i+1}} * Pr(w_n) $$

# In[83]:


words_count = Counter(TOKEN)


# ### ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# In[84]:


from functools import reduce


# In[85]:


from operator import add, mul


# In[86]:


words_count.most_common(100)


# In[87]:


reduce(mul, [1, 2, 3, 4, 5, 8])


# In[88]:


[1, 2, 3] + [3, 43, 5]


# In[89]:


frequiences = [f for w, f in words_count.most_common(100)]


# In[90]:


x = [i for i in range(100)]


# In[91]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[92]:


import matplotlib.pyplot as plt


# In[93]:


plt.plot(x, frequiences)


# In[94]:


import numpy as np


# In[95]:


plt.plot(x, np.log(frequiences))


# >NLP比较重要的规律：在很大的一个text corpus，文字集合中，出现频率第二多的单词是出现频率第一多单词频率的1/2，出现频率第n多的单词频率，是出现频率最高单词频率的1/n。

# ### ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# In[54]:


def prob_1(word):
    return words_count[word] / len(TOKEN)         # 'XX'出现次数/单词总数


# In[55]:


TOKEN[:10]


# In[56]:


# TOKEN = [str(t) for t in TOKEN]    好像没什么用


# In[57]:


TOKEN_2_GRAM = [''.join(TOKEN[i:i+2]) for i in range(len(TOKEN[:-2]))]  # 2个单词连起来


# In[58]:


TOKEN_2_GRAM[:10]


# In[59]:


words_count_2 = Counter(TOKEN_2_GRAM)    # 计算2个单词连起来的次数


# In[60]:


def prob_2(word1, word2):
    if word1 + word2 in words_count_2: 
        return words_count_2[word1+word2] / words_count[word2]
    else:
        return 1 / len(words_count)


# In[61]:


prob_2('我们', '在')


# In[62]:


prob_2('在', '吃饭')


# In[63]:


def get_probablity(sentence):
    words = list(cut(sentence))
    
    sentence_pro = 1
    
    for i, word in enumerate(words[:-1]):
        next_ = words[i+1]
        
        probability = prob_2(word, next_)
        
        sentence_pro *= probability
    
    sentence_pro *= prob_1(word[-1])
    
    return sentence_pro


# In[64]:


get_probablity('小明今天抽奖抽到一台苹果手机')


# In[65]:


get_probablity('小明今天抽奖抽到一架波音飞机')


# In[66]:


get_probablity('洋葱奶昔来一杯')


# In[67]:


get_probablity('养乐多绿来一杯')


# In[69]:


need_compared = [
    "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "今晚我去吃火锅 今晚火锅去吃我",
    "洋葱奶昔来一杯 养乐多绿来一杯"
]

for s in need_compared:
    s1, s2 = s.split()
    p1, p2 = get_probablity(s1), get_probablity(s2)
    
    better = s1 if p1 > p2 else s2
    
    print('{} is more possible'.format(better))
    print('-'*4 + ' {} with probility {}'.format(s1, p1))
    print('-'*4 + ' {} with probility {}'.format(s2, p2))


# ## 作业

# In[78]:


need_compared = [
    "早上起来去吃早饭 早饭起来去吃早上",
    "我和我的朋友一起去唱歌 朋友和唱歌一起去我",
    "我点了一杯珍珠奶茶 我点了一杯臭豆腐奶茶",
    "中国有珠穆朗玛峰 日本有珠穆朗玛峰"
]

for s in need_compared:
    s1, s2 = s.split()
    p1, p2 = get_probablity(s1), get_probablity(s2)
    
    better = s1 if p1 > p2 else s2
    
    print('{} is more possible'.format(better))
    print('-'*4 + ' {} with probility {}'.format(s1, p1))
    print('-'*4 + ' {} with probility {}'.format(s2, p2))

