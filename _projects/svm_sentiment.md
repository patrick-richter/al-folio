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

The IMDB movie review dataset contains reviews along with a positive or negative sentiment associated with each movie review (see bellow for an example). Due to the size of the whole dataset, we will only use a subset here. It contains 5,000 training samples and 1,500 test samples. Before we get started, download and save the training data [here](https://patrick-richter.github.io/assets/csv/movie_review_train.csv) and the test data [here](https://patrick-richter.github.io/assets/csv/movie_review_test.csv) (make sure that you set the format to `Page Source`).

```
Sentiment: "negative"
Review: "There's a thin line between being theatrical and being just plain forced. Forced acting. Forced takes. Forced plot. Even forced photography."
```
<br/><br/>
### **Prerequirements**

The project is written in Python and, before starting, make sure to install and import the following libraries.


<br/><br/>
### **Data Preprocessing**
