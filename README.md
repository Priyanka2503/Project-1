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
Hyperparameter tuning is done by building a hyperdrive service using Jupyter notebook.First I initialize the azure machine learning workspace, then created a compute cluster to run the experiments on and check for existing cluster. Now the existing cluster is found so  it was used instead of creating a new cluster.For this Logistic Regression algorithm was used.This is based on hyperparameters such as -C(Inverse of Regularization Strength) and -max_iter(Maximum number of iterations to converge).
ps = RandomParameterSampling({
    "--C" : uniform(0.5,1.0),
    "--max_iter" : choice(50,100,150,200) })
    
The sampling method I used is RandomSampling.It supports early termination of low-performance runs.For this a scikit-learn estimator for the training script and a HyperDriveConfig was created.The best saved model provided the accuracy of 0.9132793820121391.

![Screenshot (310)](https://user-images.githubusercontent.com/75804779/102646863-ca48e800-418a-11eb-88ec-c3ee84835147.png)

## Benefits of parameter sampler
The sampling method I used is RandomSampling.The advantage of using this is that it helps to avoid bias.It also helps in choosing the best hyperparameters and optimize for speed versus accuracy. It supports both discrete and continuous values. It supports early termination of low-performance runs. In Random Sampling, the values are selected randomly from a defined search space.

## Benefits of early stopping policy
With the help of Early Termination policy, we can terminate poorly performing runs.Here I used Bandit Policy.Bandit Policy is based on slack factor and evaluation interval. This policy will terminate runs whose primary metric is not within the specified slack factor.The bandit policy helped to avoid burning up a lot of resources while trying to find an optimal parameter, it terminates any run that does not fall within the slack factor's range.

## AutoML
The best model obtained through AutoML is VotingEnsemble.Accuracy of this is 0.9170.The Voting Ensemble estimates multiple base models and uses voting to combine the individual predictions to arrive at the final ones.
Prameters that are set for AutoMLConfig are:
automl_config = AutoMLConfig(experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    compute_target='hyper-drive',
    n_cross_validations=5)
    
The iterations of pipelines are as follows.

![Screenshot (312)](https://user-images.githubusercontent.com/75804779/102646873-ce750580-418a-11eb-9c65-e44da98d029d.png)

![Screenshot (309)](https://user-images.githubusercontent.com/75804779/102646554-3ecf5700-418a-11eb-8356-25d1c28861a6.png)

The best run was given by:
Run(Experiment: automl_azure,
Id: AutoML_81a035fb-8b11-43ad-ad89-2a9ad3abbd72_21,
Type: azureml.scriptrun,
Status: Completed)

The best model was given by:
Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=None, enable_feature_sweeping=None,
                                 feature_sweeping_config=None,
                                 feature_sweeping_timeout=None,
                                 featurization_config=None, force_text_dnn=None,
                                 is_cross_validation=None,
                                 is_onnx_compatible=None, logger=None,
                                 observer=None, task=None, working_dir=None)),
                ('prefittedsoftvotingclassifier',...
                                                                                                    min_samples_split=0.2442105263157895,
                                                                                                    min_weight_fraction_leaf=0.0,
                                                                                                    n_estimators=10,
                                                                                                    n_jobs=1,
                                                                                                    oob_score=False,
                                                                                                    random_state=None,
                                                                                                    verbose=0,
                                                                                                    warm_start=False))],
                                                                     verbose=False))],
                                               flatten_transform=None,
                                               weights=[0.18181818181818182,
                                                        0.09090909090909091,
                                                        0.09090909090909091,
                                                        0.18181818181818182,
                                                        0.09090909090909091,
                                                        0.09090909090909091,
                                                        0.09090909090909091,
                                                        0.18181818181818182]))],
         verbose=False)
Y_transformer(['LabelEncoder', LabelEncoder()])

## Pipeline comparison
The accuracy obatined through hyperdrive run is 0.9132793820121391 whereas the accuracy obtained through AutoML is 0.9170 by VotingEnsemble.The hyperdrive used logistic regression algorithm and provide its accuracy while the AutoML used several learning algorithms to obtain its accuracy.In hyperparameter tuning we are able to use only one algorithm whereas in AutoML, different models were used. As of result of this, we were able to choose the best performing model instead of sticking to just one type.Hence we are getting better accuracy in case of AutoML.Also, the time taken by hyperdrive model was 931.1sec and that by AutoML was 3309.2 sec.

## Removal of Compute Cluster

## Future work
Some of the imporvements that can be done in this are:
1.Better tuning of hyperparameters can be done.We can improve the Parameter sampler.
2. We can use of other classification algorithms in hyperdrive run.
3.We can try using other Early Termination policy or optimize the current one to get better results.
