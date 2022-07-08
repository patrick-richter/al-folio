---
layout: page
title: 'Double DQN Applied to Lunar Lander'
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
<div class="video-container"><iframe src="https://www.youtube.com/embed/0HuI1QLOCJM?rel=0&amp;controls=0&amp;showinfo=0&amp;autoplay=1&loop=1" frameborder="0" allow="autoplay; encrypted-media"></iframe></div>
<br/><br/>
### **Dataset**

