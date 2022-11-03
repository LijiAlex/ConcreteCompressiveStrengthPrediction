# ConcreteCompressiveStrengthPrediction

## Problem Statement

The quality of concrete is determined by its compressive strength, which is measured using a conventional crushing test on a concrete cylinder. The strength of the concrete is also a vital aspect in achieving the requisite longevity. It will take 28 days to test strength, which is a long period. A lot of time and effort can be saved by using Data Science to estimate compressive strength of concrete for the given mixture composition.

## Business Goal

Save waiting time on strength test. 

## Model Design

* Supervised 
* Multivariate 
* Cluster Classification
* Outlier Removal

## Performance Measure

* R<sup>2
* RMSE

## Salient Features

* Follows CICD pipeline.
* Clusterification to decide upon best model for each cluster.
* Uses GridSearchCv to find the best model.
* Maintains entity and artifact folder.
* Use seperate thread for training purpose.
* Integrated UI.
* Logs for debugging.

### Pipeline Components

* Data Ingestion
* Data Validation
* Data Transformation
* Model Training
* Model Evaluation
* Model Push

### Exception
ConcreteException: Custom exception for the project




