# learn-SentimentAnalysis

Five classes of sentiment level, which are represented by 0 to 4 in the code, respectively:
“very negative (−−)”, “negative (−)”, “neutral”, “positive (+)”, “very positive (++)”

## 1. Train a softmax classifier with the word vectors

* 可以只使用出现频率较高的(比如top10)，需要训练的是word vector前面的系数，以及截距（在softmax classifier中使用线性函数）。这种方法保留了一些word vector的信息，但丢掉了词频的信息，也丢掉了词汇排列的信息。

* A simple way of representing a sentence is taking the average of the vectors of the words in the sentence. 然后训练这个word vector前面的系数，以及截距（在softmax classifier中使用线性函数）。Averaging word vectors destroys word vectors and word order, and doesn’t handle negation.

* To avoid overfitting to the training examples and generalizing poorly to unseen examples, introduce regularization when doing classification (in fact, most machine learning tasks).

* There word vectors work better: higher dimensional word vectors may encode more information, vectors that were trained on a much larger corpus

* 测试结果可以用confusion matrix的heatmap来展示

## 2. K nearest neighbor by EMD of word vectors

类似[topic-classifier](https://github.com/mediaProduct2017/topic-classifier)

这种方法保留了一些word vector和词频的信息，但还是丢掉了词汇排列的信息。
