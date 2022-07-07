---
layout: page
title: 'CNN: Age Estimation and Gender Classification'
description: 
img: assets/img/30.jpg
importance: 1
category: Deep Learning
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/3.jpg" title="Faces" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<br/><br/>

One of the most important applications of Deep Learning is **Computer Vision**. Today, we will take a deeper dive into this topic and deploy a **Convolutional Neural Network (CNN)** to predict the age and the gender of faces from the **UTKFace dataset**.
<br/><br/>
### **Dataset**

The **UTKFace dataset** is a large-scale face dataset with an **age span from 0 to 116 years old** with a resolution of 128x128. However, to avoid long training durations, we will only use a **subset of 5,000 images**. You can download the dataset by clicking this [Link](https://patrick-richter.github.io/assets/zip/train_test.zip) (full dataset can be found on [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)).

The project is written in **Python** and, before starting, make sure to install and import the following **libraries**.

<script src="https://gist.github.com/patrick-richter/c31f5464e7cfae8f723d758591e4766d.js"></script>

To give you brief idea of how the data looks like, we will first **visualise a couple of images**. The labels regarding age and gender are hidden in the file name. For instance, the person with the file path `train_test/28_1_0_20170117180708809.jpg.chip.jpg` is 28 years old and is female (1 = female, 0 = male).

<script src="https://gist.github.com/patrick-richter/0f2dfb96c162191dc4c3b5ab40f2ed8b.js"></script>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/31.jpg" title="Visualisation" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Visualisation of 20 faces
</div>
<br/><br/>
### **Data Preprocessing**





<br/><br/>
### **CNN Model with Word Embedding**

A **Word Embedding** is a learned representation for text analysis – typically in the form of a vector – where words that are closer in the vector space are expected to be close in meaning. The representation of words is learned based on the usage of words, allowing words that are used in similar ways to result in having similar representations, naturally capturing their meaning.

There are several methods how you can implement Word Embedding. For this task, the word embedding is **implemented into a Neural Network in form of a layer**. In the training process, the embedding layer's weights are updated to best represent each of the words as a vector. This approach will learn an embedding both targeted to the specific text data (in that case, car reviews) and to the classification task.

The Neural Network is built by using **keras**. After the embedding and convolutional layer, the model also consists of two fully connected layers for the classification.

<script src="https://gist.github.com/patrick-richter/485975ee6d7114e7a29f29f932b04a01.js"></script>

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_12 (Embedding)     (None, 550, 120)          411000    
_________________________________________________________________
conv1d_12 (Conv1D)           (None, 543, 32)           30752     
_________________________________________________________________
max_pooling1d_12 (MaxPooling (None, 271, 32)           0         
_________________________________________________________________
flatten_12 (Flatten)         (None, 8672)              0         
_________________________________________________________________
dense_24 (Dense)             (None, 10)                86730     
_________________________________________________________________
dense_25 (Dense)             (None, 1)                 11        
=================================================================
Total params: 528,493
Trainable params: 528,493
Non-trainable params: 0
_________________________________________________________________
```
<br/><br/>
### **Model Training**

Now, the model only needs to be trained with `model.fit()`. Subsequently, the model can predict the sentiments for both the training and the test data.

<script src="https://gist.github.com/patrick-richter/24f55c357f2e006062ae15212612db94.js"></script>

```
Epoch 1/4
35/35 - 1s - loss: 0.6933 - accuracy: 0.5176 - val_loss: 0.6896 - val_accuracy: 0.5668
Epoch 2/4
35/35 - 0s - loss: 0.6468 - accuracy: 0.6615 - val_loss: 0.6058 - val_accuracy: 0.7292
Epoch 3/4
35/35 - 0s - loss: 0.3906 - accuracy: 0.8588 - val_loss: 0.4516 - val_accuracy: 0.8303
Epoch 4/4
35/35 - 1s - loss: 0.1246 - accuracy: 0.9674 - val_loss: 0.4976 - val_accuracy: 0.8303
```
<br/><br/>
### **Results**

The model achieves an **83.03 % accuracy** on the test data, which is quite an impressive result, considering the small dataset. With even more data, perhaps, the **Word Embedding could have even been more effective**.

<script src="https://gist.github.com/patrick-richter/f29f9144a52f1e054c8c3f71b9e8b325.js"></script>

```
The Word Embedding CNN achieves a 99.0 % accuracy on training data and a 83.03 % accuracy on test data.
```

The **confusion matrix** demonstrates that the classification is well balanced (similar false positive and false negative rate).

<script src="https://gist.github.com/patrick-richter/493b94d11c303d0f624cf7cc280c22b1.js"></script>

<div class="row">
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/13.jpg" title="Confusion matrix" class="img-fluid"%}
    </div>
</div>
<div class="caption">
    Confusion matrix (top left = true negatives, top right = false positives, bottom left = false negatives, and bottom right = true positives)
</div>
