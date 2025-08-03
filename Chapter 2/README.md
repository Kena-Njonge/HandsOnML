This file exists to document what I learned in the chapter and other details about the implemented code and notebooks.

# Summary

In Chapter 2 we deal with an end-to-end machine learning project. We learn how to deal with null features using imputers, how to scale data using scalers and standardizers, and how to correctly validate our models using cross-validation.

We look at the big picture and explore the considerations that have to be made when dealing with a real-world dataset. We explore every step that one needs to take when working on an ML project, including:

- data visualization and basic exploration  
- data cleaning and handling missing values  
- transformation steps like turning longitude and latitude into clusters  
- different types of encoding  
- feature engineering and augmentation

We also get a feel for:

- different metrics and when to use them  
- tools and functions from `sklearn`  
- how to write clean, modular code using pipelines and column transformers  
- how to validate your model properly using k-fold cross-validation  
- how to fine-tune hyperparameters using `GridSearchCV` and `RandomizedSearchCV`

Géron also gives us a few tips on long-term model support and maintaining pipelines, which I noted, but don’t expect to apply yet, more of a “good to know” for later.

# MelbourneDataset

I tried to work through a similar dataset and problem myself. The model that I created yielded an MAE of 300k USD in the task of predicting the price of a home in Melbourne.  

I believe this performance can be improved on significantly, but the purpose of this notebook was mainly to:

- get familiar with the tools  
- try out different scalers and encoders  
- practice writing custom transformers  
- and package it all into a working pipeline

In the future, I should definitely come back and address what’s lacking here as I become more comfortable with `sklearn`.

# StandardClone

Here I was completing the exercise from the book to clone some of the functionality of `StandardScaler` from `sklearn.preprocessing`.  

A great programming exercise—both for design and thinking through how to implement something cleanly using just `numpy`. It was also helpful to test everything properly and get used to a bit more structure.
