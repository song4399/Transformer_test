from hyperparams import Hyperparams as hp

import numpy as np
import codecs
import regex
import random


def load_vocab(language): # 加载词汇表文件  # 读取指定的词汇表文件，并将每行第一个提取出来，保存在vocab中
    assert language in ["cn", "en"]
    vocab = [
        line.split()[0]
        for line in codecs.open("./preprocessed/{}.txt.vocab.tsv".format(language), "r", "utf-8").read().splitlines()
        if int(line.split()[1]) >= hp.min_cnt
    ]
    # codecs.open("./{}.txt.vocab.tsv".format(language), "r", "utf-8")以指定的编码方式打开一个文件。并返回一个文件对象
    # splitlines() 将字符串按行分割成多个字符串组成的列表
    word2idx = {word: idx for idx, word in enumerate(vocab)} # 创建单词和索引的对应关系
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word # 返回索引映射 每个中文/英文单词对应的索引值


def load_cn_vocab():
    word2idx, idx2word = load_vocab("cn")
    return word2idx, idx2word


def load_en_vocab():
    word2idx, idx2word = load_vocab("en")
    return word2idx, idx2word

def create_data(source_sents, target_sents): # 用于将源语句和目标语句转换为索引序列 source_sents ['admadnam asdjksl adksld adsjlk', 'admadnam adsjlk'] 
    cn2idx, idx2cn = load_vocab("cn") # 返回索引映射 每个中文/英文单词对应的索引值
    en2idx, idx2en = load_vocab("en") # 返回索引映射 每个中文/英文单词对应的索引值

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents): # 把每个中/英文句子，转换为索引序列，并保存在新的变量中
        x = [
            cn2idx.get(word, 1) for word in (source_sent + " <eos>").split() 
        ]  # 1: OOV, </S>: End of Text # 语句末尾加上一个空格和末尾标记</S>，根据空格分割转换为列表，再获取对应的单词的索引，组成x
        # print(x) # [19, 9, 931, 4, 161, 6, 484, 1696, 2267, 4, 161, 36, 5, 161, 2723, 21, 4209, 2351, 100, 4, 2092, 635, 3]
        y = [en2idx.get(word, 1) for word in ("<bos> " + target_sent + " <eos>").split()] # 英文语句索引序列
        if max(len(x), len(y)) <= hp.maxlen-1: # 如果最大长度在范围内，就接收这个序列以及源和目标句子
            x_list.append(np.array(x)) # 句子对应的索引序列并保存在x_list中
            y_list.append(np.array(y))
            Sources.append(source_sent) # 对应的句子，还没有padding的
            # print(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32) # 创建空的数组，大小为（样本数量*最大长度）
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad( # padding！！！，
            x, [0, hp.maxlen - len(x)], "constant", constant_values=(2, 2) # [0, hp.maxlen - len(x)]指明在序列的开头和结尾分别填充的3的数量
        )
        # print(X[i])
        # [   0 1441    7  256   58  223   88 4371 1948 3056   29  296  304   36   20  701  136   18   10  825    1    2    2    2    2    2    2    2
        # 2    2    2    2    2    2    2    2    2    2    2    2    2    2    2    2    2    2    2    2    2    2]
        Y[i] = np.lib.pad(
            y, [0, hp.maxlen - len(y)], "constant", constant_values=(2, 2)
        )

    return X, Y, Sources, Targets # X Y 是np数组，不同行为不同样本，已经padding后的，每行为样本的索引序列 Sources, Targets为接收的源和目标句子


def load_data(data_type): # 加载句子seq
    if data_type == "train":
        source, target = hp.source_train, hp.target_train
    elif data_type == "test":
        source, target = hp.source_test, hp.target_test
    assert data_type in ["train", "test"]
    cn_sents = [
        regex.sub("[^\s\p{L}']", "", line) # 用于将给定字符串 line 中的非空格、字母和撇号字符替换为空字符串。
        # "[^...]" 匹配不在方括号内的任意字符
        # \s匹配任意空白换行，结果：空格和换行都保留
        # \p{L}匹配任意字母字符，结果：字母和汉字都保留
        for line in codecs.open(source, "r", "utf-8").read().split("\n") # 以utf-8读取训练/测试集文件，按照换行符来拆分成可迭代对象
        if line and line[0] != "<"
    ] # 一个大列表，里面每个元素都是一个句子，例如'我军 诸 军兵种 的 状态 '
    # print(cn_sents)
    en_sents = [
        regex.sub("[^\s\p{L}']", "", line)
        for line in codecs.open(target, "r", "utf-8").read().split("\n")
        if line and line[0] != "<"
    ]

    X, Y, Sources, Targets = create_data(cn_sents, en_sents) # 加载训练集/测试集的句子
    return X, Y, Sources, Targets


def load_train_data(): # 加载训练集数据
    X, Y, Sources, Targets = load_data("train")
    return X, Y, Sources, Targets


def load_test_data():
    X, Y, Sources, Targets = load_data("test")
    return X, Y, Sources, Targets


def get_batch_indices(total_length, batch_size): # total_length一共多少样本，batch_size批次大小
    assert (
        batch_size <= total_length
    ), "Batch size is large than total data length. Check your data or change batch size."
    current_index = 0
    indexs = [i for i in range(total_length)] # 0~total_length-1的索引列表
    random.shuffle(indexs) # 随机打乱索引表
    while 1:
        if current_index + batch_size >= total_length:
            break
        yield indexs[current_index : current_index + batch_size], current_index
        current_index += batch_size # 当前索引+批大小


# load_data("test")