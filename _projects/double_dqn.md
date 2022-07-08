---
layout: page
title: 'Double DQN Applied to Lunar Lander'
description: 
img: assets/img/3.jpg
importance: 2
category: Reinforcement Learning
---

<video autoplay>
    <source src="https://patrick-richter.github.io/assets/vid/lunarlander.mp4" type="video/mp4">
</video>

<video width="320" height="240" controls autoplay>
    <source src=”http://techslides.com/demos/sample-videos/small.ogv” type=video/ogg>
</video>

### **Dataset**

Today, we will have a look at the **AG's News Topic Classification Dataset**. The dataset includes **title and description** of news articles for **120,000 training samples** and **7,600 test samples**. Each of those is classified into of the categories **"World"**, **"Sports"**, **"Business"**, or **"Sci/Tech"** (see [here](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) for more information). Here you can see a random sample of the dataset:

```
Class: "Sci/Tech"
Title: "E-mail scam targets police chief",
Description: "Wiltshire Police warns about "phishing" after its fraud squad chief was targeted."
```

We will implement a **Random Forest classifier** to learn this problem and hopefully achieve a good result on the test dataset. Before we get started, download and save the training data [here](https://patrick-richter.github.io/assets/csv/train.csv) and the test data [here](https://patrick-richter.github.io/assets/csv/test.csv) (make sure that you set the format to `Page Source`).
<br/><br/>
### **Prerequirements**

The project is written in **Python** and, before starting, make sure to install and import the following **libraries**.

<script src="https://gist.github.com/patrick-richter/99ac21582db0a17a0c517b972aad5b85.js"></script>
<br/><br/>
### **Data Preprocessing**

To be able to preprocess the data, the **csv file** is first of all **read with pandas** and transformed into a numpy array.

<script src="https://gist.github.com/patrick-richter/3d7f631daf32e7bde0de6cdfbc535995.js"></script>

The next step is to **remove all stop words** (words such as "the", "I", or "he" that occur so frequently that they are deemed irrelevant for the classification), **digits**, and **punctuation**. By using a pre-existing stop word list from the nltk library, the undesirable words can be easily discarded.

<script src="https://gist.github.com/patrick-richter/d8c164e531a5a04e6a44ea6e03289b95.js"></script>

Following that, the remaining words are stemmed, i.e., they are reduced to their base word or stem so that similar words are being represented by the same stem word. For example, ‘leaking’ and ‘leaks’ would both be converted to ‘leak’ and would therefore be seen as the same word in the upcoming steps. Due to better performance, the **Porter Stemmer**, which is slightly less aggressive than the alternative Snowball Stemmer, was implemented.

<script src="https://gist.github.com/patrick-richter/5f01be94de49aff396ae156e9e84e387.js"></script>

Finally, to make our data processable for the Random Forest classifier, we need to vectorise the text. There are several ways how to represent text in a vector. One of the most commonly used approaches that is also used here, is to count the occurrence of words. This **count vectoriser** approach returns a vector that counts how often each of the words in the vocabulary (all different words in the training data) occur for each sample.

The problem we are facing here is that there are just **too many different words** in the training data (>20,000) which would make it incredibly computational expensive. Therefore, the `max_features` parameter is here **limited to 4,000**, i.e., only the 4,000 most frequently occurring words are considered.

<script src="https://gist.github.com/patrick-richter/cfecf2c99c43520dff84d45b794b2982.js"></script>
<br/><br/>
### **Random Forest Classifier**

Before we go into Random Forests themselves, it is important to understand how **decision trees** work. As you can see in the basic example bellow, decision trees always split in a way that the resulting subsamples are as dissimilar as possible. In other words, the tree decides to **split the feature where it can gain the most information**.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/14.jpg" title="Decision tree" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Simple decision tree (source: <a href="https://towardsdatascience.com/understanding-random-forest-58381e0602d2">Towards Data Science</a>)
</div>

A **Random Forest** is an **ensemble of multiple decision trees**, combining their predictions the wisdom of many to provide a more robust prediction (see figure below). To ensure that the trees are as uncorrelated as possible, Random Forest modifies the decision trees in two ways to add more randomness. **Bagging**, the first method, refers to the process that each tree is trained with a random subset of samples, instead of the whole dataset. **Feature Randomness** introduces randomness by only letting the decision trees pick from a subset of features at each point. Due to the highly uncorrelated trees and the prediction by committee, Random Forests are far more accurate than decision trees alone in their prediction.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/15.jpg" title="Random Forest" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Random Forest (source: <a href="https://towardsdatascience.com/understanding-random-forest-58381e0602d2">Towards Data Science</a>)
</div>
<br/><br/>
### **Implementation**

For this project, we will not implement the **Random Forest classifier** from scratch, but instead use the already implemented function from `scikit-learn`. If you are interested in checking out how to code the algorithm from scratch, check [here](https://tonyalgo.com/machinelearning/randomforest).

The Random Forest classifier has two really important **hyperparameters**. `max_depth` determines how many nodes each decision tree can maximally have. From playing around with the hyperparameter, I achieved the best results with 2,000 (half of the vocabulary). The second import hyperparameter is `n_estimators`, the number of decision trees in the Random Forest. Generally, the more trees you have, the better the results will be. However, as you can see in the figure bellow, the test accuracy only improves slightly after 50 trees. In my code, I choose quite a high number of trees (300), but this will also take you more than 30 minutes to train.

<script src="https://gist.github.com/patrick-richter/cb7e3035227c9d02e9fc35e2afd8214f.js"></script>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/17.jpg" title="Number of trees" class="img-fluid" %}
    </div>
</div>
<br/><br/>
### **Results**

By comparing the predictions with the real labels, we obtain the **training and test accuracy**:

<script src="https://gist.github.com/patrick-richter/981eaa67dfdba061b41c637ead27ebab.js"></script>

```
Training accuracy: 99.95 %
Test accuracy: 89.11 %
```

The achieved **test accuracy of almost 90 %** is quite impressive, especially considering that the Random Forest classifier is a comparatively simple algorithm in contrast to the Deep Neural Networks that would otherwise be used for such tasks. Moreover, by using count vectoriser as the vector representation, any context in the sentences is lost. If we had used a CNN for example, we could have implemented better representations such as Word Embedding. The current benchmark (they used a CNN) by [Zhang et al. (2015)](https://papers.nips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf) for this dataset is 92 %.

When looking at the **confusion matrix**, we can see that most of the prediction errors originate from mixing up the "Business" and "Sci/Tech" categories. "Sports", as you would expect, is quite distinguishable and has the highest prediction accuracy with 97.2 %.

<script src="https://gist.github.com/patrick-richter/8615eb731b829098957ed27f49a7641b.js"></script>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/16.jpg" title="Confustion matrix" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Confusion Matrix
</div>
