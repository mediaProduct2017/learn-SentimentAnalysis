# learn-SentimentAnalysis

Five classes of sentiment level, which are represented by 0 to 4 in the code, respectively:
“very negative (−−)”, “negative (−)”, “neutral”, “positive (+)”, “very positive (++)”

## 1. Train a softmax classifier with the word vectors

* 可以只使用出现频率较高的word(比如top10)，需要训练的是word vector前面的系数，以及截距（在softmax classifier中使用线性函数）。这种方法保留了一些word vector的信息，但丢掉了词频的信息，也丢掉了词汇排列的信息。

* 如果word vector用的是固定的值，就只有系数W(:)需要拟合，总的维度是Cd，C是类别的个数，d是feature（word vector）的个数，这与一般的机器学习的数据拟合是类似的。如果word vector也需要拟合，需要拟合的变量的总的维度就是Cd+Vd（总维度极高，很容易过拟合），C是类别的个数，V是word vector的维度，d是word vector的个数。If you only have a small training data set, don’t train the word vectors. If you have have a very large dataset, it may work beZer to train word vectors to the task.

* A simple way of representing a sentence is taking the average of the vectors of the words in the sentence. 然后训练这个word vector前面的系数，以及截距（在softmax classifier中使用线性函数）。Averaging word vectors destroys word vectors and word order, and doesn’t handle negation.

* To avoid overfitting to the training examples and generalizing poorly to unseen examples, introduce regularization when doing classification (in fact, most machine learning tasks). Regularization目标是使每个系数都很小，从而使得预测的函数曲线很平滑，提高test的准确度。在拟合过程中，常出现的问题就是过拟合，也就是参数的增加等使得training error不断减小，但到了一定程度，test error反而不断增加，这意味着出现了过拟合，模型的预测能力相当有限。

* These word vectors work better: higher dimensional word vectors may encode more information, vectors that were trained on a much larger corpus

* 测试结果可以用confusion matrix的heatmap来展示

![Cost function and softmax regression](images/softmax.png)

对于此处的sentiment analysis，C个类别，d个feature（比如d个word vector）。W.x是线性函数的矩阵形式。

对于某些预测word的自然语言处理，C个word，每个word vector的维度是d。

x本质上是一个set（一个集合，哪怕表面是用list表示的），x的各个元素并没有严格的先后顺序，也就不能抓住相应的信息。比如对于图像识别，x是一组像素点，如果使用basic neural network来做识别，x中并没有包含这些像素点的排列信息。

## 2. K nearest neighbor by EMD of word vectors

类似[topic-classifier](https://github.com/mediaProduct2017/topic-classifier)

这种方法保留了一些word vector和词频的信息，但还是丢掉了词汇排列的信息。

## 3. Simple single word classification

![word sentiment classification](images/sentiment.png)
