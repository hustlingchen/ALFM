Aspect-Aware Latent Factor Model (ALFM)
========

This is our implementation for the paper:

<b>Zhiyong Cheng, Ying Ding, Lei Zhu, Mohan Kankanhalli. [Aspect-Aware Latent Factor Model:  Rating Prediction with Ratings and Reviews.](https://dl.acm.org/citation.cfm?id=3186145)  In Proceedings of WWW '18, Lyon, France, April 23-27, 2018.</b>

Our model is worked in two steps: in the first step, a topic model is used to extract the topic representation of aspects; in the second step, the resutls of the topic model are integrated into an aspect-aware latent factor model to estimate users and items latent factors as well as factor weights.
 
 <b>Please cite our WWW'18 paper if you use our codes. Thanks!</b>
 
 Author: Dr. Zhiyong Cheng (https://sites.google.com/view/zycheng)

Codes
==
Our code is in java: 

<b>"topicmodel"</b> package: the implementation of the topic model: 
  --  please run "tuningAspectNumberandTopicNumber.java"  to get the results by setting data path correctly
 
<b>"alfm"</b> package: the implementation of the aspect-aware latent factor model
 -- please run “topicFactorTuning.java" to get the results. 
 
 Notice that alfm is implemented based on ["LibRec"](https://www.librec.net/). We already include the necessary 'jar' packages in the 'lib' fold
 
 Examples:
 ==
 In the codes, we also put three datasets which were used in our experiments: "Beauty", "Digital Music", "Music Instruments" in the <b> "data" </b> fold
 
 The model can be tested on the three datasets by running "tuningAspectNumberandTopicNumber.java" frist and then “topicFactorTuning.java"
 
 The topic results (based on 5 aspects and 5 topics) are saved in "model/topicmodel/"

 Results of "alfm" are saved into "model/alfm/"
