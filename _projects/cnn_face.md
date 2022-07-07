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
        {% include figure.html path="assets/img/30.jpg" title="Faces" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

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

Before we can start with preprocessing, the data needs to be stored in a pandas `DataFrame`. Then we can split the data into **training and test data**, with 20 % being used for testing purposes.

<script src="https://gist.github.com/patrick-richter/99b7a7ccd9da4ef629594623d9b96cc8.js"></script>

```
                                              filename   age  gender
0     train_test/86_1_0_20170120225751953.jpg.chip.jpg  86.0       1
1     train_test/26_1_0_20170116171048641.jpg.chip.jpg  26.0       1
2     train_test/52_0_1_20170117161018159.jpg.chip.jpg  52.0       0
3     train_test/16_0_0_20170104003740977.jpg.chip.jpg  16.0       0
4     train_test/27_0_3_20170119210058457.jpg.chip.jpg  27.0       0
...                                                ...   ...     ...
4995  train_test/86_1_2_20170105174813405.jpg.chip.jpg  86.0       1
4996  train_test/28_0_2_20170107212142294.jpg.chip.jpg  28.0       0
4997   train_test/1_1_0_20170109194452834.jpg.chip.jpg   1.0       1
4998  train_test/54_0_0_20170109010040814.jpg.chip.jpg  54.0       0
4999  train_test/52_0_3_20170119200211340.jpg.chip.jpg  52.0       0

[5000 rows x 3 columns]
```

**Data Augmentation** is an enormously useful tool that is used in most Computer Vision projects. By applying certain variations such as rotation, zoom, or a horizontal flip to the images, it artificially creates new training data from already existing training data

It is so popular mainly due to two big advantages. First, it helps you **getting more training** data without having to mine new data. Secondly, it is a key tool to **prevent overfitting**. The model can never fit perfectly on the training data because each epoch you will have slight variations of each image.

The easiest way of implementing Data Augmentation is with the `ImageDataGenerator` from `tensorflow`. Here, we apply rotation, zoom, and a horizontal flip to our images. Moreover, the rgb value range is scaled to 0 and 1.

<script src="https://gist.github.com/patrick-richter/139d7e751e09ed73c299f2c379c952a3.js"></script>

```
Found 4000 validated image filenames.
Found 1000 validated image filenames.
```
<br/><br/>
### **CNN Model Construction**

As we are facing a **muti-label problem**, we need to construct a CNN model that outputs two different values. `keras` offers a way to split the model in two at any point of the Neural Network. 

Here, two of the three convolutional layers (including padding) are shared between the two branches. After the second convolutional layer, the age and the gender prediction **split into two branches**. Each of the branches include one more convolutional layer, one dense layer, and one output layer. As you can see below, dropout is also employed in each of the branches and in the shared-learning part.

<script src="https://gist.github.com/patrick-richter/138cd717b5c78177db74175152a015d8.js"></script>

<div class="row">
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/33.jpg" title="Model architecture" class="img-fluid"%}
    </div>
</div>
<div class="caption">
    Model architecture
</div>
<br/><br/>
### **Model Training**

The following step, obviously, is to **train the model**. Since the total loss is the sum of the age and the gender loss, we need to make sure that the values of the two loss functions are balanced. Therefore, we weigh the `binary_crossentropy` loss of the gender classification 500 times higher than the `mse` of the age estimation. The model is trained for 60 epochs.

<script src="https://gist.github.com/patrick-richter/603f720753c5e045a8b626d11480f5c3.js"></script>

```
Epoch 1/60
200/200 [==============================] - 15s 73ms/step - loss: 862.1603 - dense_age_loss: 521.0098 - dense_gender_loss: 0.6823 - dense_age_mae: 17.5374 - dense_gender_accuracy: 0.6021 - val_loss: 689.9260 - val_dense_age_loss: 393.1925 - val_dense_gender_loss: 0.5935 - val_dense_age_mae: 14.5779 - val_dense_gender_accuracy: 0.6960
Epoch 2/60
200/200 [==============================] - 14s 72ms/step - loss: 664.3820 - dense_age_loss: 365.3248 - dense_gender_loss: 0.5981 - dense_age_mae: 14.6565 - dense_gender_accuracy: 0.6846 - val_loss: 592.0073 - val_dense_age_loss: 328.2217 - val_dense_gender_loss: 0.5276 - val_dense_age_mae: 13.6442 - val_dense_gender_accuracy: 0.7430

...

Epoch 59/60
200/200 [==============================] - 15s 73ms/step - loss: 275.9303 - dense_age_loss: 138.7035 - dense_gender_loss: 0.2745 - dense_age_mae: 8.8752 - dense_gender_accuracy: 0.8787 - val_loss: 294.9626 - val_dense_age_loss: 132.8631 - val_dense_gender_loss: 0.3242 - val_dense_age_mae: 8.2970 - val_dense_gender_accuracy: 0.8750
Epoch 60/60
200/200 [==============================] - 15s 74ms/step - loss: 266.4663 - dense_age_loss: 129.6572 - dense_gender_loss: 0.2736 - dense_age_mae: 8.6207 - dense_gender_accuracy: 0.8847 - val_loss: 278.6100 - val_dense_age_loss: 122.3719 - val_dense_gender_loss: 0.3125 - val_dense_age_mae: 8.1059 - val_dense_gender_accuracy: 0.8770
```
<br/><br/>
### **Results**

On the test data, the model achieves an **8.11 MAE on age estimation** and an **87.70 % accuracy on gender classification**. This is a good result, considering the small dataset and the fact that two different variables are predicted in one model.

When we plot the **training curves**, we can see that due to dropout and data augmentation there is **no over- or underfitting** present.

<script src="https://gist.github.com/patrick-richter/3b25d563450db7b877f50099a7d1c7b7.js"></script>

<div class="row">
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/34.jpg" title="Training performance" class="img-fluid"%}
    </div>
</div>
<div class="caption">
    Training performance
</div>

Finally, to give you an idea of how the predictions look like with images present, I sampled 20 images and printed out the prediction with the actual label in brackets. Especially the age estimation would be quite hard even for humans.

<script src="https://gist.github.com/patrick-richter/dc92047f522be9562f6f2e8bdfc791e0.js"></script>

<div class="row">
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/32.jpg" title="Predictions" class="img-fluid"%}
    </div>
</div>
<div class="caption">
    20 Faces with prediction and actual age and gender in brackets
</div>
