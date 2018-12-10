---
title: Automatic Playlist Recommender
---
## CS 109 A Final Project
CS 109 A Final Project

>Group 54: Yalda Amini, João Araujo, Daniel Barjum
> Dec 2018



**Project question is as follows:**

“Given a random sample of the tracks from a playlist(s), can we predict the class of the remaining songs in a given playlist(s)?”

Our general strategy will be to obtain information about a random sample of tracks within a playlist, attempt to classify that set of songs based on the variables presented above (and/or others), and then predict the remaining set of songs in a given playlist based on our model(s). Given the classification approach that we are taking to this unsupervised case (by classifying songs into 5 different genres), we will test models such as logistic regression (one vs. many), decision trees, LDA, QDA, etc. 

Our training set will be the random sample of tracks, our y values will be the most common class (genre) of the random sample, defined by a threshold. The same logic will apply to our test set, which will be the remaining tracks inside a playlist. With this logic, we will be able to test the performance of our models on the training and test sets, perform cross-validations and define hyperparameters (if any). Our decision of a best model will be defined based on these performance exercises, and will take into account computational complexity and the trade-off between performance and interpretability.

For this EDA our y values are defined based on a classification of the genre of each song in a playlist - detailed explanation in “Additional Feature Engineering (classification)”. We plan to explore more complex set ups in the next phase of the project. One way we plan to do so is by classifying songs into indexes that comprise multiple characteristics of songs in a playlist, besides the genre. 
