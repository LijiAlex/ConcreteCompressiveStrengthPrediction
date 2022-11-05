# ConcreteCompressiveStrengthPrediction

## Problem Statement

The quality of concrete is determined by its compressive strength, which is measured using a conventional crushing test on a concrete cylinder. The strength of the concrete is also a vital aspect in achieving the requisite longevity. It will take 28 days to test strength, which is a long period. A lot of time and effort can be saved by using Data Science to estimate compressive strength of concrete for the given mixture composition.

## Business Goal

Save waiting time on strength test. 

## Proposed Design

Supervised prediction system that estimates the strength of the concrete given the mixture composition

## Tech stack used
* Python
* Flask
* ML algorithms
* Heroku

## Training pipeline
![Training Pipeline](https://user-images.githubusercontent.com/59106185/200120386-2d8a2da6-1c1c-4fee-a710-a17421ed4d70.jpg)

## Salient Features

* Follows CICD pipeline.
* Clusterification to decide upon best model for each cluster.
* Uses GridSearchCv to find the best model.
* Maintains entity and artifact folder.
* Use seperate thread for training purpose.
* Integrated UI.
* Logs for debugging.
* Custom exception.

## Feature engineering steps

* Outlier removal
* Normalization
* Standardization
* Clusterification

## Performance Measure

* R<sup>2
* RMSE








