---
layout: page
title: 'Random Forest: News Article Classification'
description: 
img: assets/img/3.jpg
importance: 2
category: Classic Machine Learning
---


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/3.jpg" title="News" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<br/><br/>
### **Dataset**

Today, we will have a look at the AG's News Topic Classification Dataset. The dataset includes title and description of news articles for 120,000 training samples and 7,600 test samples. Each of those is classified into of the categories "World", "Sports", "Business", or "Sci/Tech" (see [here](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) for more information). Here you can see a random sample of the dataset:

```
Class: "Sci/Tech"
Title: "E-mail scam targets police chief",
Description: "Wiltshire Police warns about "phishing" after its fraud squad chief was targeted."
```

We will implement a Random Forest classifier to learn this problem and hopefully achieve a good result on the test dataset. Before we get started, download and save the training data [here](https://patrick-richter.github.io/assets/csv/train.csv) and the test data [here](https://patrick-richter.github.io/assets/csv/test.csv) (make sure that you set the format to `Page Source`).
<br/><br/>
### **Prerequirements**

The project is written in Python and, before starting, make sure to install and import the following libraries.

<script src="https://gist.github.com/patrick-richter/99ac21582db0a17a0c517b972aad5b85.js"></script>
<br/><br/>
### **Data Preprocessing**

To be able to preprocess the data, the csv file is first of all read with pandas and transformed into a numpy array.

<script src="https://gist.github.com/patrick-richter/3d7f631daf32e7bde0de6cdfbc535995.js"></script>

The next step is to remove all stop words (words such as "the", "I", or "he" that occur so frequently that they are deemed irrelevant for the classification), digits, and punctuation. By using a pre-existing stop word list from the nltk library, the undesirable words can be easily discarded.

<script src="https://gist.github.com/patrick-richter/d8c164e531a5a04e6a44ea6e03289b95.js"></script>

Following that, the remaining words are stemmed, i.e., they are reduced to their base word or stem so that similar words are being represented by the same stem word. For example, ‘leaking’ and ‘leaks’ would both be converted to ‘leak’ and would therefore be seen as the same word in the upcoming steps. Due to better performance, the Porter Stemmer, which is slightly less aggressive than the alternative Snowball Stemmer, was implemented.

<script src="https://gist.github.com/patrick-richter/5f01be94de49aff396ae156e9e84e387.js"></script>

Finally, to make our data processable for the Random Forest classifier, we need to vectorise the text. There are several ways how to represent text in a vector. One of the most commonly used approaches that is also used here, is to count the occurrence of words. This count vectoriser approach returns a vector that counts how often each of the words in the vocabulary (all different words in the training data) occur for each sample.

The problem we are facing here is that there are just too many different words in the training data (>20,000) which would make it incredibly computational expensive. Therefore, the `max_features` parameter is here limited to 4,000, i.e., only the 4,000 most frequently used words are considered.

<script src="https://gist.github.com/patrick-richter/cfecf2c99c43520dff84d45b794b2982.js"></script>
<br/><br/>
### **Random Forest**


<br/><br/>
### **Data Preprocessing**
