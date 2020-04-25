# KDD2020Multimodalities
KDD Cup 2020 Challenges for Modern E-Commerce Platform: Multimodalities Recall

Baseline Score: 0.5538

方案详解：https://zhuanlan.zhihu.com/p/135984016 

https://fasttext.cc/docs/en/english-vectors.html   `crawl-300d-2M.vec.zip`    

https://nlp.stanford.edu/projects/glove/  `glove.840B.300d.zip`

https://github.com/google-research/bert `BERT-Base, Uncased`


数据全部存放在data目录下

运行顺序

1.preprocess.py(只读了1w的数据,要读取更多数据,谨慎修改函数`pd.read_csv`的参数，有大内存机器的忽略)

2.[image-concat-query]-wwm_uncased_L12-768_v3_quart.ipynb

