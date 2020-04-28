# NLPlay

## What is NLPlay?
NLPlay is a repository centralizing implementations of key NLP algorithms ,to tackle Text Classification, Sentiment Analysis & Question Answering problems.
The idea is to have a collection of ready to use algorithms & building blocks , to allow people to perform rapid benchmarks over standard datasets or their own ones.  

## Supported models & features

### Python/Sklearn (CPU Only)
-  **TFIDF / BOW + linear Model** : [A statistical interpretation of term specificity and its application in retrieval - 1972](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.8343&rep=rep1&type=pdf)
-  **FastText**   : [Bag of Tricks for Efficient Text Classification - 2016](https://arxiv.org/abs/1607.01759)
-  **NBSVM**      : [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification- 2012](https://www.aclweb.org/anthology/P12-2018.pdf)

### Pytorch (CPU/GPU)
-  **CNN**        : [Convolutional Neural Networks for Sentence Classification - 2014](https://arxiv.org/abs/1607.01759)
-  **FastText**   : [Bag of Tricks for Efficient Text Classification - 2016](https://arxiv.org/abs/1607.01759)
-  **DAN**        : [Deep Unordered Composition Rivals Syntactic Methods for Text Classification - 2015](https://arxiv.org/abs/1607.01759)
-  **NBSVM++**    : [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification- 2012](https://www.aclweb.org/anthology/P12-2018.pdf)
-  **QRNN**       : [Quasi-Recurrent Neural Networks - 2016](https://arxiv.org/pdf/1611.01576)
-  **LSTM/BiLSTM**: [Long Short Term Memory - 1997](https://www.bioinf.jku.at/publications/older/2604.pdf),
                    [Neural architectures for named entity recognition - 2016](https://arxiv.org/pdf/1603.01360.pdf)
-  **GRU/BiGRU**  : [Neural Machine Translation by Jointly Learning to Align and Translate - 2014](https://arxiv.org/pdf/1409.0473)

## Additional Pytorch Optimizers
-  **AdaBound**  : [Adaptive Gradient Methods with Dynamic Bound of Learning Rate - 2019](https://arxiv.org/pdf/1902.09843)
-  **DiffGrad**  : [diffGrad: An Optimization Method for Convolutional Neural Networks - 2019](https://arxiv.org/pdf/1909.11015)
-  **Lookahead** : [Lookahead Optimizer: k steps forward, 1 step back - 2019](https://arxiv.org/pdf/1907.08610)
-  **RAdam**     : [On the Variance of the Adaptive Learning Rate and Beyond - 2020](https://arxiv.org/pdf/1908.03265)
-  **Ranger**    : [An Adaptive Remote Stochastic Gradient Method for Training Neural Networks - 2019](https://arxiv.org/pdf/1905.01422)
## Additional Pytorch Activation Functions
-  **Mish**           : [Mish: A Self Regularized Non-Monotonic Neural Activation Function - 2019](https://arxiv.org/pdf/1908.08681)
-  **Swish/SwishPlus**: [Flatten-T Swish: a thresholded ReLU-Swish-like activation function for deep learning - 2019](https://arxiv.org/ftp/arxiv/papers/1812/1812.06247.pdf)
-  **LightRelu**      : [LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks - 2019](https://arxiv.org/abs/1901.05894)
-  **Threshold Relu** : [An improved activation function for deep learning - Threshold Relu, or TRelu - 2019](https://github.com/lessw2020/TRelu)
## Additional Pytorch loss
-  **FocalLoss** : [Focal Loss for Dense Object Detection - 2017](https://arxiv.org/pdf/1708.02002)
## Datasets
-  **Sentiment analysis**      : IMDB, MR
-  **Question classification** : TREC6, TREC50
-  **Text classification**     : 20 newsgroups, AGNews, Amazon Review Polarity, Amazon Review Full , DBpedia, Yelp Review Polarity, Yelp Review Full, Sogou News, Yahoo Answers 

## Examples

## Tutorials

## Todo / Next Steps:
1. Include additional Models :
    -  **HAN**   : [Hierarchical Attention Networks for Document Classification - 2016](https://www.aclweb.org/anthology/N16-1174.pdf)
    -  **SIF**   : [A Simple but Tough-to-Beat Baseline for Sentence Embeddings - 2016](https://openreview.net/forum?id=SyK00v5xx)
    -  **USIF**  : [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline - 2018](https://www.aclweb.org/anthology/W18-3012.pdf)
    -  **RE2**   : [Simple and Effective Text Matching with Richer Alignment Features - 2019](https://arxiv.org/pdf/1908.00300)
    -  **BiMPM** : [Bilateral Multi-Perspective Matching for Natural Language Sentences - 2017](https://arxiv.org/pdf/1702.03814)

2. Include additional Datasets :
    -  **SNLI**    : [The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)
    -  **Quora**   : [The Quora question pairs dataset](https://www.kaggle.com/c/quora-question-pairs)
    -  **SciTail** : [A Textual Entailment Dataset from Science Question Answering](https://allenai.org/data/scitail)
    -  **WikiQA**  : [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/YangYihMeek_EMNLP-15_WikiQA.pdf)

3. Others :
    -  Include [Nvidia Apex - Mixed Precision](https://github.com/NVIDIA/apex) to improve GPU memory footprint
    -  Include Cross validation mechanism
    -  Include Metrics (F1,AUC...) + Confusion Matrix
    -  Include [Microsoft NNI](https://github.com/microsoft/nni) for Hyper Parameters Tuning 
    -  Include [MLflow](https://www.mlflow.org/docs/latest/index.html#) for Experiments tracking
