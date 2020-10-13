# NLPlay

## What is NLPlay?
NLPlay is a toolbox / repository, centralizing implementations of key NLP algorithms in one place,to tackle Text Classification, Sentiment Analysis & Question Answering problems.
The idea is to have a collection of ready to use algorithms & building blocks , to allow people to quickly benchmark/customize those different model architectures, over standard datasets or their own ones.  

## Supported models & features

### Python/Sklearn (CPU Only)
-  **TFIDF Ngrams + SGD linear Model** : [A statistical interpretation of term specificity and its application in retrieval - 1972](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.8343&rep=rep1&type=pdf)
-  **FastText**   : [Bag of Tricks for Efficient Text Classification - 2016](https://arxiv.org/abs/1607.01759)
-  **NBSVM**      : [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification - 2012](https://www.aclweb.org/anthology/P12-2018.pdf)

### Pytorch (CPU/GPU)
-  **FastText**   : [Bag of Tricks for Efficient Text Classification - 2016](https://arxiv.org/abs/1607.01759)
-  **DAN**        : [Deep Unordered Composition Rivals Syntactic Methods for Text Classification - 2015](https://arxiv.org/abs/1607.01759)
-  **MLP**        : A model based on an embedding layer and a configurable pooling & feed-forward neural network on top
-  **NBSVM++**    : [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification - 2012](https://www.aclweb.org/anthology/P12-2018.pdf) - Source : [FastAI](https://github.com/fastai/fastai/blob/release-1.0.61/old/fastai/nlp.py) 
-  **CharCNN**    : [Character-level Convolutional Networks for Text Classification - 2015](https://arxiv.org/pdf/1509.01626.pdf) 
-  **TextCNN**    : [Convolutional Neural Networks for Sentence Classification - 2014](https://arxiv.org/pdf/1408.5882.pdf) - Source : [Galsang](https://github.com/galsang/CNN-sentence-classification-pytorch)
-  **EXAM**       : [Explicit Interaction Model towards Text Classification - 2018](https://arxiv.org/pdf/1811.09386.pdf) - !UNDER DEVELOPMENT!
-  **DPCNN**      : [Deep Pyramid Convolutional Neural Networks for Text Categorization - 2017](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf) - Source : [Cheneng](https://github.com/Cheneng/DPCNN/blob/master/model/DPCNN.py)
-  **QRNN**       : [Quasi-Recurrent Neural Networks - 2016](https://arxiv.org/pdf/1611.01576) - Source : [Dreamgonfly](https://github.com/dreamgonfly/deep-text-classification-pytorch)
-  **SWEM**       : [Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms - 2018](https://arxiv.org/pdf/1805.09843.pdf)
-  **SRU**        : [Simple Recurrent Units for Highly Parallelizable Recurrence - 2017](https://arxiv.org/pdf/1709.02755.pdf) - Source : [Asappresearch](https://github.com/asappresearch/sru)
-  **LSTM/BiLSTM**: [Long Short Term Memory - 1997](https://www.bioinf.jku.at/publications/older/2604.pdf),
                    [Neural architectures for named entity recognition - 2016](https://arxiv.org/pdf/1603.01360.pdf)
-  **GRU/BiGRU**  : [Neural Machine Translation by Jointly Learning to Align and Translate - 2014](https://arxiv.org/pdf/1409.0473)

## Additional Pytorch Optimizers
-  **AdaBound**  : [Adaptive Gradient Methods with Dynamic Bound of Learning Rate - 2019](https://arxiv.org/pdf/1902.09843) - Source : [Luolc](https://github.com/Luolc/AdaBound)
-  **DiffGrad**  : [diffGrad: An Optimization Method for Convolutional Neural Networks - 2019](https://arxiv.org/pdf/1909.11015) - Source : [Less Wright](https://github.com/lessw2020/Best-Deep-Learning-Optimizers/tree/master/diffgrad)
-  **Lookahead** : [Lookahead Optimizer: k steps forward, 1 step back - 2019](https://arxiv.org/pdf/1907.08610) - Source : [lonePatient](https://github.com/lonePatient/lookahead_pytorch)
-  **QHAdam**    : [Quasi-hyperbolic momentum and Adam for deep learning - 2019](https://arxiv.org/pdf/1810.06801.pdf) - Source : [FacebookResearch](https://github.com/facebookresearch/qhoptim)
-  **RAdam**     : [On the Variance of the Adaptive Learning Rate and Beyond - 2020](https://arxiv.org/pdf/1908.03265) - Source : [LiyuanLucasLiu](https://github.com/LiyuanLucasLiu/RAdam)
-  **Ranger**    : [An Adaptive Remote Stochastic Gradient Method for Training Neural Networks - 2019](https://arxiv.org/pdf/1905.01422) - Source : [Less Wright](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
## Additional Pytorch Activation Functions
-  **Mish**           : [Mish: A Self Regularized Non-Monotonic Neural Activation Function - 2019](https://arxiv.org/pdf/1908.08681) - Source : [Diganta Misra](https://github.com/digantamisra98/Mish)
-  **Swish/SwishPlus**: [Flatten-T Swish: a thresholded ReLU-Swish-like activation function for deep learning - 2019](https://arxiv.org/ftp/arxiv/papers/1812/1812.06247.pdf) - Source : [Geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/activations/activations.py)
-  **LiSHT/LightRelu**: [LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks - 2019](https://arxiv.org/abs/1901.05894) - Source : [Less Wright](https://github.com/lessw2020/LightRelu)
-  **Threshold Relu** : [An improved activation function for deep learning - Threshold Relu, or TRelu - 2019](https://github.com/lessw2020/TRelu) - Source : [Less Wright](https://github.com/lessw2020/TRelu)
## Additional Pytorch loss
-  **FocalLoss**          : [Focal Loss for Dense Object Detection - 2017](https://arxiv.org/pdf/1708.02002) - Source : [mbsariyildiz](https://github.com/mbsariyildiz/focal-loss.pytorch)
-  **LabelSmoothingLoss** : [Rethinking the Inception Architecture for Computer Vision - 2015](https://arxiv.org/pdf/1512.00567.pdf) - Source : [OpenNMT](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/loss.py)
-  **Supervised Contrastive Loss**: [Supervised Contrastive Learning - 2020](https://arxiv.org/pdf/2004.11362.pdf) - Source : [Yonglong Tian](https://github.com/HobbitLong/SupContrast)

## Datasets
-  **Sentiment analysis**      : [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/), [MR](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
-  **Question classification** : [TREC6, TREC50](https://trec.nist.gov/data/qa.html)
-  **Text classification**     : [20 newsgroups](http://qwone.com/~jason/20Newsgroups/), [AGNews](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), [Amazon Review Polarity, Amazon Review Full](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) , [DBpedia](https://wiki.dbpedia.org/Datasets), [Yelp Review Polarity, Yelp Review Full](https://www.yelp.com/dataset), [Sogou News](http://wwwconference.org/www2008/papers/pdf/p457-wang.pdf), [Yahoo Answers](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l) 
## Others
- [**parlib**](https://github.com/jeremypoulain/nlplay/blob/master/nlplay/utils/parlib.py)    : Parallel Processing for large lists (ie corpus pre-processing), Pandas DataFrames or Series, using [joblib](https://joblib.readthedocs.io/en/latest/)
- [**DSManager / WordVectorsManager**](https://github.com/jeremypoulain/nlplay/blob/master/nlplay/data/cache.py) : Automatic reference and download of key datasets & pretrained vectors (Glove, FastText...)

## Examples

## Tutorials

## Todo / Next Steps:
1. Include additional Models :
    -  **HAN**          : [Hierarchical Attention Networks for Document Classification - 2016](https://www.aclweb.org/anthology/N16-1174.pdf)
    -  **SIF**          : [A Simple but Tough-to-Beat Baseline for Sentence Embeddings - 2016](https://openreview.net/forum?id=SyK00v5xx)
    -  **USIF**         : [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline - 2018](https://www.aclweb.org/anthology/W18-3012.pdf)
    -  **RE2**          : [Simple and Effective Text Matching with Richer Alignment Features - 2019](https://arxiv.org/pdf/1908.00300)
    -  **BiMPM**        : [Bilateral Multi-Perspective Matching for Natural Language Sentences - 2017](https://arxiv.org/pdf/1702.03814)
    -  **MaLSTM/MaGRU** : [Siamese Recurrent Architectures for Learning Sentence Similarity - 2016](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023)

2. Include additional Datasets :
    -  **SNLI**    : [The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)
    -  **Quora**   : [The Quora question pairs dataset](https://www.kaggle.com/c/quora-question-pairs)
    -  **SciTail** : [A Textual Entailment Dataset from Science Question Answering](https://allenai.org/data/scitail)
    -  **WikiQA**  : [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/YangYihMeek_EMNLP-15_WikiQA.pdf)

3. Others :
    -  <s>Include [Nvidia Apex - Mixed Precision](https://github.com/NVIDIA/apex) to improve GPU memory footprint on Turing/Volta/Ampere architectures</s>
    -  Include support of [Google TPU](https://cloud.google.com/tpu/docs/tpus) for training & inference via [PyTorch/XLA](https://github.com/pytorch/xla)
    -  Include Cross validation mechanism
    -  Include Metrics (F1,AUC...) + Confusion Matrix
    -  Include automatic [EDA](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15) reporting features
    -  Include a [streamlit](https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace) app to easily explore & debug model predictions errors and identify potential root causes (ie tokenization, unseen tokens, sentence length,class confusion..)  
    -  Include [Microsoft NNI](https://github.com/microsoft/nni) for Hyper Parameters Tuning ([TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf),
     [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf), [Hyperband](https://arxiv.org/pdf/1603.06560.pdf), [BOHB](https://www.automl.org/blog_bohb/)... )
    -  Include [MLflow](https://www.mlflow.org/docs/latest/index.html#) for Experiments tracking
