---
layout: page
title: Sentiment Classification with CNN and Word Embedding
description: 
img: assets/img/12.jpg
importance: 1
category: Deep Learning
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/12.jpg" title="Ford image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


The data used in this project is a large dataset of Ford motor vehicle reviews. Each of the 1,382 reviews is labelled with a positive or negative sentiment. The goal of this project is to train a Convolutional Neural Network (CNN) with Word Embedding that can correctly predict the sentiment of a review. You can download the dataset by clicking this [Link](https://patrick-richter.github.io/assets/csv/car_reviews.csv) and subsequently saving the csv file. To give you brief idea of how these reviews look like, here is an example of a negative sentiment review:

```
"In 1992, we bought a new Taurus and we really loved it. 
So in 1999, we decided to try a new Taurus. 
I did not care for the style of the newer version but bought it anyway. 
I do not like the new car half as much as I liked our other one."
```

<br/><br/>
### Prerequirements

The project is written in Python and before starting, make sure to install and import the following libraries.

<script src="https://gist.github.com/patrick-richter/f5935b8651b1fce5a54aa279fe21ff88.js"></script>


<br/><br/>
### Data Preprocessing

To be able to preprocess the data, the csv file is first of all read with pandas and transformed into a numpy array. Then, all negative reviews receive the label 0, whereas all positive reviews receive 1 as their label.

<script src="https://gist.github.com/patrick-richter/b64f6ea0eeb602649fa1253937fcaedd.js"></script>

The next step is to remove all stop words (words such as "the", "I", or "he" that occur so frequently that they are deemed irrelevant for the classification), digits, and punctuation. By using a pre-existing stop word list from the nltk library, the undesirable words can be easily discarded.

<script src="https://gist.github.com/patrick-richter/060239691b03d428c3c7cba00ffd3333.js"></script>

Following that, the remaining words are stemmed, i.e., they are reduced to their base word or stem so that similar words are being represented by the same stem word. For example, ‘leaking’ and ‘leaks’ would both be converted to ‘leak’ and would therefore be seen as the same word in the upcoming steps. Due to better performance, the Porter Stemmer, which is slightly less aggressive than the alternative Snowball Stemmer, was implemented.

<script src="https://gist.github.com/patrick-richter/289a51d0b99af9cff7d46ca6717cfb5e.js"></script>

To be able to later assess the model with no bias, it is paramount to split the dataset into a training dataset that is used for training, and a test dataset that serves for unbiased performance assessment. Here, 20% of the dataset is used for testing. Also note that it is important to set a random state, making sure that we always have the same split, even when you run it multiple times.

<script src="https://gist.github.com/patrick-richter/c8078f707316a5310f9fc3d27fd5434f.js"></script>

For the Word Embedding approach, it is necessary to have a vocabulary of all words that occur in the training dataset. However, it has been shown that generally Word Embeddings where the vocabulary has been reduced to only the most predictive words, yield better results and train significantly faster due to the reduced data.

One common approach to limit the vocabulary is by assuming that the words that occur very rarely are less predictive than the words with higher occurrence. Here, it is decided (through trial and error) that only words that occur 5 times or more often are included into the vocabulary. This reduces the vocabulary from 9,033 to 3,424 different words.

<script src="https://gist.github.com/patrick-richter/c833741c218588ff1c5f2f0fcddbaebc.js"></script>

As the final step of preprocessing, the reviews must be encoded. In other words, the text (now only consisting of the words that are included in the vocabulary) is converted to a sequence where each integer represents one word in the vocabulary. Bellow, you can see an exemplary sequence.

Furthermore, the maximum number of words of a sequence is limited to prevent the input data from getting too large. The sequences are padded to a length of 550, meaning all words that exceed the limit are cut off.

<script src="https://gist.github.com/patrick-richter/e36a5bc1acee398f32f26943a8de18a2.js"></script>

```
[  38  818    3 2253  200    1  997 1042  682 1064 2503  978   33  654
   56   14 2867    3  176   79    2   65  388  683  890    3  298  162
  425  551 1064  139   65  819  710  503   44  103  267   13  197   78
    1  425  140 3109  269   10   11    1 1065    4  336   32  270  367
   74   36   15 1188   96  234   94   33  123  128 1775   65  235  127
  645 1670  383  276   82   52   65  276 2868  105   47  637  405  172
    9    5   57   23  405  203   45  336   21  103  717  130  100   75
    1  458   40  446  105  405   39  919 1117   15  360  600 2145  165
   63   46   52 3110  516    1  299  433  121  520  276  216  156   16
 2146   56    1  228   21   92  130   56  228   21  119  561 1027    1
 1013 1488  855 1013   24  150 1343   76  119 1489  611 2254   56    5
  222  791 1411  105 1118  497 1343 3111  170  208  222  556  689 1103
  112  492  351  127   31  535    4   98  202   60   36   96  276   12
   70  110  856 1490 1066    5  258   65   44  202   71  731  319  153
   12   65   44   65  156  417 1064  337   66  132   76 1223  308  163
 1344  902  128 1566    7   26  776 2147   26  107  267  158  605  277
  167  536 2869 2504  638  470  446  510  284   13  110 1142 3112  523
  342  271   94   76  207  185   37  216  189  156   16  383  710  116
  276   59  312   51  101  335    3  101   21   18  477  386  581  267
   17 1567  934    7  269  168   13  200  121    2  792    2  159  295
  295  168  810 1064  601  148  144  190  102  320 2373   41 1067 1715
   45  261   57  102  261   35  312   35 1671   65   16  140 3109  396
   17   16  127   99  168   11   31  170  122   28   97   29   40    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0]
```
<br/><br/>

### CNN Model with Word Embedding

A Word Embedding is a learned representation for text analysis – typically in the form of a vector – where words that are closer in the vector space are expected to be close in meaning. The representation of words is learned based on the usage of words, allowing words that are used in similar ways to result in having similar representations, naturally capturing their meaning.

There are several methods how you can implement Word Embedding. For this task, the word embedding is implemented into a Neural Network in form of a layer. In the training process, the embedding layer's weights are updated to best represent each of the words as a vector. This approach will learn an embedding both targeted to the specific text data (in that case, car reviews) and to the classification task.

The Neural Network is built by using keras. After the embedding and convolutional layer, the model also consists of two fully connected layers for the classification.

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
### Model Training

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
# SAFsdfsdfsd

Every project has a beautiful feature showcase page.
It's easy to include images in a flexible 3-column grid format.
Make your photos 1/3, 2/3, or full width.

To give your project a background in the portfolio page, just add the img tag to the front matter like so:

    ---
    layout: page
    title: project
    description: a project with a background image
    img: /assets/img/12.jpg
    ---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, *bled* for your project, and then... you reveal it's glory in the next row of images.


<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>


The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}
```html
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
```
{% endraw %}

# Title

## Hallo

### Hallo

#### Hallo
