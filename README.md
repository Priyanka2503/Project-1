# Optimizing a ML Pipeline in Azure
## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.This model is then compared to an Azure AutoML run.
## Summary
The provided dataset is a Bank Marketing dataset.We have to predict whether a client subscribed a term deposit or not. The best performing model was a VotingEnsemble obtained through AutoML with the primary metric 'accuracy' value as 0.9170.
The main steps are shown in diagram below:

![Image](https://video.udacity-data.com/topher/2020/September/5f639574_creating-and-optimizing-an-ml-pipeline/creating-and-optimizing-an-ml-pipeline.png)

## Scikit-learn Pipeline
At first the train.py file was uploaded to the Notebooks in Azure.After that we had imported the bank marketing dataset csv file using the TabularDatasetFactory.
Data is located at: "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

Then the dataset is split into training(80%) and test(20%) data.Then the scikit-learn logistic regression model is created for testing the accuracy.
model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

## Hyperparameter tuning
Hyperparameter tuning is done by building a hyperdrive service using Jupyter notebook.First I initialize the azure machine learning workspace, then created a compute cluster to run the experiments on and check for existing cluster. Now the existing cluster is found so  it was used instead of creating a new cluster.For this Logistic Regression algorithm was used.
The sampling method I used is RandomSampling.It supports early termination of low-performance runs.For this a scikit-learn estimator for the training script and a HyperDriveConfig was created.The best saved model provided the accuracy of 0.9132793820121391.

![Screenshot (309)](https://user-images.githubusercontent.com/75804779/102646554-3ecf5700-418a-11eb-8356-25d1c28861a6.png)

## Benefits of parameter sampler
The sampling method I used is RandomSampling.The advantage of using this is that it helps to avoid bias.It also helps in choosing the best hyperparameters and optimize for speed versus accuracy. It supports both discrete and continuous values. It supports early termination of low-performance runs. In Random Sampling, the values are selected randomly from a defined search space.

## Benefits of early stopping policy
With the help of Early Termination policy, we can terminate poorly performing runs.Here I used Bandit Policy.Bandit Policy is based on slack factor and evaluation interval. This policy will terminate runs whose primary metric is not within the specified slack factor.The bandit policy helped to avoid burning up a lot of resources while trying to find an optimal parameter, it terminates any run that does not fall within the slack factor's range.

## AutoML
The best model obtained through AutoML is VotingEnsemble.Accuracy of this is 0.9170.
The iterations of pipelines are as follows.
****************************************************************************************************
ITERATION   PIPELINE                                       DURATION      METRIC      BEST
         0   MaxAbsScaler LightGBM                          0:00:55       0.9153    0.9153
         1   MaxAbsScaler XGBoostClassifier                 0:01:06       0.9151    0.9153
         2   MaxAbsScaler RandomForest                      0:00:49       0.8942    0.9153
         3   MaxAbsScaler RandomForest                      0:00:53       0.8880    0.9153
         4   MaxAbsScaler RandomForest                      0:00:49       0.8130    0.9153
         5   MaxAbsScaler RandomForest                      0:00:53       0.7548    0.9153
         6   SparseNormalizer XGBoostClassifier             0:01:11       0.9129    0.9153
         7   MaxAbsScaler GradientBoosting                  0:00:55       0.9039    0.9153
         8   StandardScalerWrapper RandomForest             0:00:49       0.9005    0.9153
         9   MaxAbsScaler LogisticRegression                0:00:57       0.9086    0.9153
        10   MaxAbsScaler ExtremeRandomTrees                0:02:27       0.8880    0.9153
        11   SparseNormalizer XGBoostClassifier             0:01:11       0.9122    0.9153
        12   MaxAbsScaler LightGBM                          0:00:50       0.8924    0.9153
        13   MaxAbsScaler LightGBM                          0:01:04       0.9045    0.9153
        14   SparseNormalizer XGBoostClassifier             0:02:13       0.9140    0.9153
        15   StandardScalerWrapper LightGBM                 0:00:53       0.8953    0.9153
        16   StandardScalerWrapper RandomForest             0:01:14       0.8880    0.9153
        17   StandardScalerWrapper LightGBM                 0:00:55       0.8880    0.9153
        18   StandardScalerWrapper ExtremeRandomTrees       0:01:01       0.8880    0.9153
        19   StandardScalerWrapper LightGBM                 0:00:45       0.9073    0.9153
        20   SparseNormalizer XGBoostClassifier             0:00:52       0.9114    0.9153
        21   MaxAbsScaler LightGBM                          0:00:50       0.8884    0.9153
        22   SparseNormalizer LightGBM                      0:00:55       0.9065    0.9153
        23    VotingEnsemble                                0:01:27       0.9170    0.9170
        24    StackEnsemble                                 0:01:38       0.9148    0.9170
****************************************************************************************************

![Image](C:\Users\priyanka umre\Pictures\Screenshots)

## Pipeline comparison
The accuracy obatined through hyperdrive run is 0.9132793820121391 whereas the accuracy obtained through AutoML is 0.9170 by VotingEnsemble.The hyperdrive used logistic regression algorithm and provide its accuracy while the AutoML used several learning algorithms to obtain its accuracy.In hyperparameter tuning we are able to use only one algorithm whereas in AutoML, different models were used. As of result of this, we were able to choose the best performing model instead of sticking to just one type.Hence we are getting better accuracy in case of AutoML.

## Future work
Some of the imporvements that can be done in this are:
1.Better tuning of hyperparameters can be done.We can improve the Parameter sampler.
2. We can use of other classification algorithms in hyperdrive run.
3.We can try using other Early Termination policy or optimize the current one to get better results.
