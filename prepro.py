from hyperparams import Hyperparams as hp
import codecs
import os
import regex
from collections import Counter


def make_vocab(fpath, fname): # 统计词频，并进行预处理得到preprocessed文件
    """Constructs vocabulary. # 构建词汇
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    """
    text = codecs.open(fpath, "r", "utf-8").read() # 读取所有的句子
    text = regex.sub("[^\s\p{L}']", "", text)
    words = text.split() # 根据空格和换行分割字符
    word2cnt = Counter(words)
    if not os.path.exists("preprocessed"):
        os.mkdir("preprocessed")
    with codecs.open("preprocessed/{}".format(fname), "w", "utf-8") as fout:
        fout.write(
            "{}\t1000000003\n{}\t1000000002\n{}\t1000000001\n{}\t1000000000\n".format(
                "<bos>", "<eos>", "<pad>", "<unknown>", 
            )
        )
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == "__main__":
    make_vocab(hp.source_train, "cn.txt.vocab.tsv")
    make_vocab(hp.target_train, "en.txt.vocab.tsv")
    print("Done")
