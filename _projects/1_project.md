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

### Prerequirements

The project is written in Python and before starting, make sure to install and import the following libraries.

<script src="https://gist.github.com/patrick-richter/f5935b8651b1fce5a54aa279fe21ff88.js"></script>


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
