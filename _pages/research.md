---
layout: page
permalink: /research/
title: Research
description:
nav: true
nav_order: 4
---

##### **Bachelor's Thesis: "Development of a Data-Driven Battery Aging Model"**

**Abstract:**

Predicting battery aging for lithium-ion batteries is crucial, because of this component's exceptional impact on the overall electric vehicle's durability. Since it often takes months or years to run experiments examining battery aging, battery aging models can safe plenty of development time. Conventional models often find it difficult to depict the complex, nonlinear aging mechanisms. Battery aging models based on Machine Learning have shown promising results. In this work, a data-based Long Short-Term Memory network (LSTM) model is developed, ultilizing a publicly available dataset of 174 lithium-ion batteries. The LSTM's hyperparameters are optimized with a grid search, whereby the best model achieves a mean average error (MAE) of only 0.33 % in predicting the course of the state of health (SOH). It predicts the SOH profile of all validation samples in a 1.3 % MAE margin, with 72 % of the samples in a margin of 0.3 %. By applying the model to a real-life EV charging profile, the work advances into unchartered territory and proposes a model advancement to overcome the problems encountered.

[**`Download`**](https://patrick-richter.github.io/assets/pdf/bachelors_thesis.pdf)
<br/><br/>
##### **Master's Dissertation: "Object Detection for Autonomous Driving Using Deep Learning" (still ongoing)**

**Project Description:**

Accurately detecting and locating objects such as vehicles, pedestrians, cyclist, traffic lights, or street limits is of paramount importance for autonomous driving. There is no room for errors, as they can often cause deadly accidents. Early and continuous detection of objects on the street helps the car in making good and proactive decisions. 

Car manufacturers, like Mercedes-Benz and Tesla, that are already today selling cars with high levels of autonomy, all use Deep Neural Networks to identify objects in their surroundings. There are two different approaches that are widely being used. The most common one is that the image from the car’s front camera is complemented by LIDAR data. The additional data helps better assessing the image depth and therefore, better locating three-dimensional objects. However, due to the high costs of LIDAR sensors there has recently been an effort to dispense the LIDAR data and solely focus on the image data for depth estimation. Tesla is one of the major proponents of this approach, even though it is obviously harder to estimate depth and range only by using images.

The main objective of this project will be to develop a Deep Neural Network that detects objects surrounding cars with a high accuracy. Thereby, a CNN architecture will be used and customised to either the KITTI Vision Benchmark Suit or the Cityscapes Dataset, the two most common autonomous driving datasets. The first objective is to build a Deep Neural Network with a CNN architecture that can detect the objects in two dimensions with very good accuracy. In the beginning, the focus will be to keep the model as simple as possible. After that, I would like to tackle three-dimensional object detection, particularly monocular image-based methods. 

