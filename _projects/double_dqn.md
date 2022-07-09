---
layout: page
title: 'Double DQN Applied To Lunar Lander'
description: 
img: assets/img/40.jpg
importance: 2
category: Reinforcement Learning
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/40.jpg" title="Movie" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<br/><br/>
### **Lunar Lander Environment**

This project aims to train a **Double Deep Q-Learning (Double DQN)** agent to successfully land a **Lunar Lander**, run via the python library `Box2D` within Open-AI Gym.

This is a **deterministic environment** (the next state is solely determined by the action taken, no uncertainty) which provides a fully observable mixed observation space. Each observation of the **environment’s state** consists of coordinates relative to origin, velocity components and whether each leg is grounded. There four actions available to the lander – fire one of three thrusters or do nothing.

The environment is considered **’solved’** when the episode score **reaches 200 points or more**. Roughly 100-140 points are awarded for safely arriving on the landing pad, 10 points for each leg being in contract with the ground (i.e. landing upright) and an additional ±100 points for a crash landing or arrival at rest respectively. To discourage long engine fires, the lander receives -0.3 points for each frame that the main/side engine is firing. This allows the agent to first learn how to stabilise its flight and then focus on perfecting the landing itself. With a random policy, as seen in the video below, the agent reaches a score of -180.
<br/><br/>
<div class="video-container">
    <iframe src="https://www.youtube.com/embed/0HuI1QLOCJM?rel=0&amp;showinfo=0&amp;autoplay=1&loop=1" frameborder="0" allow="autoplay; encrypted-media"></iframe>
</div>
<br/><br/>
### **Prerequirements**

The project is written in **Python** and, before starting, make sure to install and import the following **libraries**.

<script src="https://gist.github.com/patrick-richter/b54284cb307ceb73ce16432b919831eb.js"></script>

<br/><br/>
### **Double DQN Agent**

<script src="https://gist.github.com/patrick-richter/5822d2a212c22477be51c5ae156c5079.js"></script>

<br/><br/>
### **Hyperparameters**

<script src="https://gist.github.com/patrick-richter/fb964f44288652ccb58ecbb0e7f6b744.js"></script>

<br/><br/>
### **Agent Training**

<script src="https://gist.github.com/patrick-richter/92f65b296cd9bb040602563f93321642.js"></script>

```
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         [(None, 8)]               0         
_________________________________________________________________
dense_6 (Dense)              (None, 128)               1152      
_________________________________________________________________
dense_7 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_8 (Dense)              (None, 4)                 516       
=================================================================
Total params: 18,180
Trainable params: 18,180
Non-trainable params: 0
_________________________________________________________________
Model: "model_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_4 (InputLayer)         [(None, 8)]               0         
_________________________________________________________________
dense_9 (Dense)              (None, 128)               1152      
_________________________________________________________________
dense_10 (Dense)             (None, 128)               16512     
_________________________________________________________________
dense_11 (Dense)             (None, 4)                 516       
=================================================================
Total params: 18,180
Trainable params: 18,180
Non-trainable params: 0
_________________________________________________________________
Episode 0 Reward -358.87 Average -358.87 Epsilon 0.99 Time 0.27 s
Episode 1 Reward -282.10 Average -282.10 Epsilon 0.99 Time 0.25 s
Episode 2 Reward -118.17 Average -227.46 Epsilon 0.99 Time 0.31 s

...

Episode 997 Reward 244.27 Average 249.01 Epsilon 0.1 Time 1.08 s
Episode 998 Reward 217.02 Average 248.45 Epsilon 0.1 Time 1.18 s
Episode 999 Reward 272.7 Average 248.39 Epsilon 0.1 Time 1.5 s
```

<script src="https://gist.github.com/patrick-richter/5e3e074d871b10377dabf908582995d0.js"></script>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/41.jpg" title="Convergence" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Convergence graph
</div>
<br/><br/>
<div class="video-container">
    <iframe src="https://www.youtube.com/embed/3Lggk1YQ61U?rel=0&amp;showinfo=0&amp;autoplay=1&loop=1" frameborder="0" allow="autoplay; encrypted-media"></iframe>
</div>
