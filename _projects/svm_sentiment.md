---
layout: page
title: 'SVM: Movie Review Sentiment Classification'
description: 
img: assets/img/1.jpg
importance: 3
category: Classic Machine Learning
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1.jpg" title="Movie" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<br/><br/>
### **Dataset**

The IMDB movie review dataset contains reviews along with a positive or negative sentiment associated with each movie review (see below for an example). Due to the size of the whole dataset, we will only use a subset here. It contains 5,000 training samples and 1,500 test samples. Before we get started, download and save the training data [here](https://patrick-richter.github.io/assets/csv/movie_review_train.csv) and the test data [here](https://patrick-richter.github.io/assets/csv/movie_review_test.csv) (make sure that you set the format to `Page Source`).

```
Sentiment: "negative"
Review: "There's a thin line between being theatrical and being just plain forced. Forced acting. Forced takes. Forced plot. Even forced photography."
```
<br/><br/>
### **Prerequirements**

The project is written in Python and, before starting, make sure to install and import the following libraries. 

<script src="https://gist.github.com/patrick-richter/589d04f8790a130d58c1f4af9244d74a.js"></script>
<br/><br/>
### **Data Preprocessing**

To be able to preprocess the data, the csv file is first of all read with pandas and transformed into a numpy array. Then, all negative reviews receive the label 0, whereas all positive reviews receive 1 as their label.

<script src="https://gist.github.com/patrick-richter/9de6c91da31ae5351c87d684c4a54276.js"></script>

The next step is to remove all stop words (words such as "the", "I", or "he" that occur so frequently that they are deemed irrelevant for the classification), digits, and punctuation. By using a pre-existing stop word list from the nltk library, the undesirable words can be easily discarded.

<script src="https://gist.github.com/patrick-richter/27a5b13e97b7e513bb9a32222697bb98.js"></script>

Following that, the remaining words are stemmed, i.e., they are reduced to their base word or stem so that similar words are being represented by the same stem word. For example, ‘leaking’ and ‘leaks’ would both be converted to ‘leak’ and would therefore be seen as the same word in the upcoming steps. Due to better performance, the Snowball Stemmer, which is slightly more aggressive than the alternative Porter Stemmer, was implemented.

<script src="https://gist.github.com/patrick-richter/29a1779474147d0c7640c24f1c6a5f97.js"></script>

Finally, to make our data processable for the SVM classifier, we need to vectorise the text. There are several ways how to represent text in a vector. One of the most commonly used approaches that is also used here, is to count the occurrence of words. However, we are not using the normal count vectorsier, and are instead implementing the Term Frequency Inverse Document Frequency (TF-IDF) vectoriser here. Rather than only returning the word count, the TF-IDF vectoriser takes into account that some words are more common in general and, therefore, weighs them less. 

<script src="https://gist.github.com/patrick-richter/dcaf4ff2ecfa3170aa9dfc641cb8d116.js"></script>
<br/><br/>
### **SVM Classifier**

A Support Vector Machines (SVM) is a supervised Machine Learning algorithm that works really well for text classification problems. It is quick and performs better than Deep Learning algorithms for a small number of samples.

In general, SVMs try to find the best hyperplane to split n-dimensional data. This approach works well for linearly seperable data. However, most real-world problems are nonlinear. This is where kernels come into play. Kernels transforms linearly inseparable data into a higher-dimensional space, making the data linearly separable again.

Finding the right kernel with the fitting hyperparameters is the main challenge when designing an SVM classifier. Fortunately, Scikit-learn offer the `GridSearchCV` class that helps finding the right hyperparameters. You can easily imput your parameter space and the algorithm will output the best hyperparameter combination. In our case, to speed up the process we are only inputting a subsample of the whole dataset.


<script src="https://gist.github.com/patrick-richter/5f9538480c04afb5d0582b56da9c9fde.js"></script>

```
{'C': 10, 'gamma': 0.1, 'kernel': 'sigmoid'} 0.844
```
<br/><br/>
### **Results**


<script src="https://gist.github.com/patrick-richter/6a1f6d10e2864f248bf3e898fd23c035.js"></script>

```
Train accuracy: 95.82 %
Test accuracy: 86.07 %
```


<script src="https://gist.github.com/patrick-richter/3363b11ec2aad3ae4907e233e5b5d610.js"></script>

<div class="row">
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/20.jpg" title="Confusion matrix" class="img-fluid"%}
    </div>
</div>
<div class="caption">
    Confusion matrix (top left = true negatives, top right = false positives, bottom left = false negatives, and bottom right = true positives)
</div>
