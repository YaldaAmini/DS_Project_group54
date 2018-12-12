---
title: Automatic Playlist Recommender
---
## CS 109 A Final Project
CS 109 A Final Project

>Group 54: Yalda Amini, João Araujo, Daniel Barjum
> Dec 2018



**Project question is as follows:**

Our general strategy was to obtain information about a random sample of tracks within a playlist, classify those songs and classify the playlist (based on the genres of the artist of the songs), to face the challenge of unsupersived learning. Then, we got features for the songs within a playlist, and colapsed those features by playlist. These collapsed variables characterize a playlist and represent our predictive features for machine learning. 

Our training and test set were a sample of playlists well distributed among the five classes. We tested models on this dataset, such as logistic regression (one vs. many), decision trees, random forest, LDA, QDA, and AdaBoost. We compared the performance of our models on the training and test sets, perform cross-validations and define hyperparameters. Our decision of our best model, random forest, was defined based on these performance exercises, and took into account computational complexity and the trade-off between performance and interpretability.

“Can we predict the class of a playlist, and use the features of its songs, to recommend appropriate new songs to that playlist?”