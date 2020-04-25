# Flight-delay-prediction

Please use the data from Kaggle: https://www.kaggle.com/usdot/flight-delays/data

## Domain Background - Air Traffic
Aerial commute is increasingly important as the globalization advances and the world population grows. However, the air traffic is also becoming a challenge, especially for the most used regional hubs. While transportation infrastructure is mainly a role for the governments, predicting the flight delays may be accessible for private initiative and it will benefit those passengers running tight on schedule by allowing them to reorganize their tasks on advance.

The most common causes of flight delays are varied. While some are not related to accessible data, others are within reach. The inaccessible data will remain as noise caused from security, maintenance and disaster issues. The accessible data are weather and congestion that hopefully will shed some light to predict some of the flight delays.

## Data Analysis and Data Processing
In order to have a better understanding of the data and have a spatial vision of the quantitative variables Exploratory Data Analysis is performed and PCA, k-means is employed for dimensionality reduction.

## Problem Statement - How much will be the flight delay?
Basically, the predictive model shall be able to answer the following question:
Given the departure information, by how many minutes will be the arrival of the flight be delayed?

## Model validation
Employed various supervised machine learninh techniques such as Decision Trees with Minimum Cost Complexity Pruning, Random Forests, Bagging with trees, Artificial Neural Network and KNN and performed hyper-parameter tuninig for optimal results
