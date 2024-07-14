# Email Similarity

## Project Overview

In this project, the task is to evaluate the performance of scikit-learn’s Naive Bayes classifier on several different datasets. By analyzing the accuracy of the classifier, the project aims to determine which datasets are more challenging to distinguish. For example, it seeks to answer how difficult it is to differentiate between emails discussing hockey and those discussing soccer, as well as how challenging it is to distinguish between emails about hockey and those about technology.

## Steps

1. **Load Datasets**: Various datasets containing emails about different topics are loaded. These datasets include categories such as hockey, soccer, and technology.

2. **Data Preprocessing**: The email datasets are preprocessed to be suitable for the Naive Bayes classifier. This involves cleaning the text data, vectorizing the text into numerical format, and splitting the data into training and testing sets.

3. **Model Training**: A Naive Bayes classifier is implemented using scikit-learn. The classifier is trained on the training sets of each dataset to learn the distinguishing features of each category.

4. **Model Evaluation**: The trained Naive Bayes classifier is tested on the testing sets. The accuracy of the classifier is recorded for each dataset pair to assess how well the classifier distinguishes between different topics.

5. **Comparison and Analysis**: The accuracies of the classifier across different dataset pairs are compared. This analysis reveals which topics are more difficult to distinguish and provides insights into the complexity of differentiating between various email topics.

## Conclusion

This project leverages scikit-learn’s Naive Bayes classifier to determine the difficulty of distinguishing between emails on various topics. By comparing classifier accuracies, it identifies which datasets present more significant challenges in terms of similarity and differentiation.
