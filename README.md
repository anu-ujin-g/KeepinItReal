# KeepinItReal
"Keepin' It Real: How to Identify Fake and Genuine Reviews"  
DSGA-1003 Machine Learning Project
## Summary of Plans
For this project, our group will use the CodaLab Fake Review dataset provided to
create a classification model for identifying fake reviews.
## Proposed Approach
We will use the pre-split training & validation sets, and use auROC & AP as the evaluation metrics. One concern with this dataset is the moderate class imbalance (~10% fake reviews). To address this concern we will use resampling via SMOTE (with both over and undersampling) in order to reach a class balance of ~50% before proceeding to feature engineering. We will then train the models using algorithms taught in the course.
## Suggested Experiments
We will first perform an EDA to see what criteria to use when classifying the reviews, ranging from sentiment analysis, lexical features such as word n-grams, part-of-speech n-grams, to semantic inconsistencies. This will allow for a common, base level of feature engineering (review length, content similarity, etc.) to be performed before transforming the reviews. The group will then use a divide and conquer approach, with each member selecting a unique featurization method and algorithm to perform the classification task. The biggest distinction between featurization implementations will come from pitting the classical Bag Of Words and CountVectorizers NL approaches against the newer pre-trained BERT and Word2Vec models. Each member will have different decisions to make based on which NL approach they take (such as stop word list selection or reinforcement training) with the end result being four unique NL featurization implementations. Once we have featurized our data, we can move onto model implementation and hyperparameter selection. We will use SVM, Logistic Regression, Neural Network, and Naive Bayes models to create classifiers for the datasets. While all datasets will start with the same search space, it’s likely they will settle on different hyperparameters. Thus each model will be trained and validated against each dataset in a bake-off, providing us with the best performing model to use on the test dataset.
## Useful Links
* [CodaLab Worksheet](https://worksheets.codalab.org/worksheets/0xbf3610354e014dc0a425cc5f49379262)
* [Google Drive](https://drive.google.com/drive/u/2/folders/1JXFNYpMoxMTjDL9esgtu1YDADvtinGdn)
* [Trello Workflow](https://trello.com/b/62i8ABdb/workflow)
* [Project Dataset](https://worksheets.codalab.org/worksheets/0x33171fbfe67049fd9b0d61962c1d05ff)
* [Piazza Project Guideline](https://piazza.com/class/k5cm3iggktn1od?cid=516)
## Timeline
- [x] 30/31 Mar: Group Meeting; Proposal Draft
- [x] 01 Apr: Proposal Due
- [x] 05 Apr: Progress Update #1, EDA
- [x] 06 Apr: Progress Update #2, Class Imbalance
- [x] 15 Apr: Feature Engineering
- [x] 20 Apr: Initial Model Building
- [x] 29 Apr: Model Comparison (ROC AUC)
- [x] 03 May: AP Scores + Hyperparameter Optimization
- [ ] 07 May: Balancing/Tuning to correct low AP
- [ ] 15 May: _goal:_ CodaLab implementation
- [ ] 15 May: **Draft Write-up**
- [ ] 19 May: Project Due

## Project Dependencies
* python3
* pandas, numpy
* sklearn
* imblearn
* spacy + 'en_trf_bertbaseuncased_lg'
* gensim + 'word2vec-google-news-300'
