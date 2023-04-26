# GWAS
Implementing GWAS (Genome wide association studies) on FHS (Framingham Heart Study) dataset 

## Aim: 

To investigate the relationship between genetic variants and blood pressure in the FHS dataset.

## Steps:

### Preprocessing:

a. Removing individuals (rows) with missing data

b. Excluding SNPs with a minor allele frequency less than 5%:

SNPs (Single Nucleotide Polymorphisms) are genetic variants that occur when there is a change in a single nucleotide (A, C, T, or G) at a specific position in the DNA sequence. 
The minor allele frequency (MAF) is the frequency of the least common allele in a population. If a SNP has a minor allele frequency less than 5%, it means that the allele that occurs less frequently is present in less than 5% of the population. 
In the given code, SNPs with a minor allele frequency less than 5% are excluded from the analysis. This is done because such SNPs are rare in the population, and their effect on the phenotype may not be well understood. Excluding these SNPs can help reduce noise and improve the accuracy of the analysis.

c. Performing imputation to fill in missing genotypes.

### GWAS using regularization methods:

Ridge, Lasso, and Elastic Net are regularization techniques used in linear regression.

 a. LASSO (Least Absolute Shrinkage and Selection Operator) 

Lasso regression is similar to Ridge regression but uses a different penalty term that results in some coefficients being exactly zero.

 b. Ridge

Ridge regression adds a penalty term to the sum of squared errors, which shrinks the coefficients towards zero and reduces the impact of multicollinearity.

 c. Elastinet

Elastic Net combines Ridge and Lasso regularization, allowing for both variable selection and reduction of multicollinearity. It uses two hyperparameters, alpha and l1_ratio, to control the balance between Ridge and Lasso regularization.
If l1_ratio = 0, ElasticNet performs Ridge regression, and if l1_ratio = 1, it performs Lasso regression.

### Analysis: 

Identifying the most important genetic variants associated with 1. systolic blood pressure (SBP) and 2. diastolic blood pressure (DBP)
Lasso performs better than ridge. Hence in elastinet the l1 ratio leans towards the lasso regression.

#### LASSO : 

i. SYSBP: 

Best parameters:  {'alpha': 0.1, 'max_iter': 1000, 'tol': 1e-05}
Best score:  0.6544035446998497
R-squared score: 0.651
Mean Squared Error: 182.127

no of features selected: 25
selected features:  

        ['RANDID', 'SEX', 'TOTCHOL', 'AGE', 'DIABP', 'CURSMOKE', 'CIGPDAY',
       'BMI', 'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP',
       'TIME', 'HDLC', 'DEATH', 'ANGINA', 'MI_FCHD', 'ANYCHD', 'CVD',
       'HYPERTEN', 'TIMEMI', 'TIMECVD', 'TIMEHYP']

ii. DIABP:

Best parameters:  {'alpha': 0.1, 'max_iter': 1000, 'tol': 0.001}
Best score:  0.5726923173742647
R-squared score: 0.550
Mean Squared Error: 55.907

no of features selected: 22
selected features:  

       ['RANDID', 'SEX', 'AGE', 'SYSBP', 'CIGPDAY', 'BMI', 'BPMEDS', 'HEARTRTE',
       'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP', 'TIME', 'HDLC', 'LDLC',
       'DEATH', 'CVD', 'HYPERTEN', 'TIMEMI', 'TIMEMIFC', 'TIMESTRK',
       'TIMEHYP']

iii. Both:  

Best parameters:  {'alpha': 0.1, 'max_iter': 1000, 'tol': 0.001}
Best score:  0.7863359687830991
R-squared score: 0.775
Mean Squared Error: 27.959

Features selected: 22 out of 39

    ['RANDID', 'SEX', 'AGE', 'SYSBP', 'CIGPDAY', 'BMI', 'BPMEDS', 'HEARTRTE',
    'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP', 'TIME', 'HDLC', 'LDLC',
    'DEATH', 'CVD', 'HYPERTEN', 'TIMEMI', 'TIMEMIFC', 'TIMESTRK',
    'TIMEHYP']

#### 2. Ridge:

i. SYSBP:

Best hyperparameters:  {'alpha': 10, 'max_iter': 1000, 'tol': 0.001}
Best performance:  0.6544313764394936
Mean squared error:  183.2885925910883
R2 score:  0.6487944951684046

Number of important features:  30
Important features:  

       ['RANDID', 'SEX', 'TOTCHOL', 'AGE', 'DIABP', 'CURSMOKE', 'CIGPDAY',
       'BMI', 'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP',
       'TIME', 'HDLC', 'LDLC', 'DEATH', 'ANGINA', 'ANYCHD', 'CVD', 'HYPERTEN',
       'TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK', 'TIMECVD',
       'TIMEDTH', 'TIMEHYP']

ii. DIABP:

Best hyperparameters:  {'alpha': 10, 'max_iter': 1000, 'tol': 0.001}
Best performance:  0.572143240850451
Mean squared error:  56.28765155098866
R2 score:  0.5469791411135423

Number of important features:  31
Important features:  

       ['RANDID', 'SEX', 'TOTCHOL', 'AGE', 'SYSBP', 'CURSMOKE', 'CIGPDAY',
       'BMI', 'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP',
       'TIME', 'HDLC', 'LDLC', 'DEATH', 'ANGINA', 'MI_FCHD', 'ANYCHD', 'CVD',
       'HYPERTEN', 'TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK',
       'TIMECVD', 'TIMEDTH', 'TIMEHYP']

iii. Both:

Best hyperparameters:  {'alpha': 10, 'max_iter': 1000, 'tol': 0.001}
Best performance:  0.7860207931189587
Mean squared error:  28.16071423492733
R2 score:  0.7734572100053749

Number of important features:  31
Important features:  

       ['RANDID', 'SEX', 'TOTCHOL', 'AGE', 'SYSBP', 'CURSMOKE', 'CIGPDAY',
       'BMI', 'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP',
       'TIME', 'HDLC', 'LDLC', 'DEATH', 'ANGINA', 'MI_FCHD', 'ANYCHD', 'CVD',
       'HYPERTEN', 'TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK',
       'TIMECVD', 'TIMEDTH', 'TIMEHYP']

#### 2. Elastinet:

i. SYSBP:

Best hyperparameters:  {'alpha': 0.01, 'l1_ratio': 0.1}
Best performance:  0.6545569058169981
Mean squared error:  183.122765432614
R2 score:  0.6491122422255643

no of features selected: 31
selected features:  

       ['RANDID', 'SEX', 'TOTCHOL', 'AGE', 'DIABP', 'CURSMOKE', 'CIGPDAY',
       'BMI', 'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP',
       'TIME', 'HDLC', 'LDLC', 'DEATH', 'ANGINA', 'MI_FCHD', 'ANYCHD', 'CVD',
       'HYPERTEN', 'TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK',
       'TIMECVD', 'TIMEDTH', 'TIMEHYP']


ii. DIABP:

Best hyperparameters:  {'alpha': 0.1, 'l1_ratio': 1.0}
Best performance:  0.5726597293133686
Mean squared error:  55.92097241975414
R2 score:  0.5499302909730657

no of features selected: 22
selected features: 

       ['RANDID', 'SEX', 'AGE', 'SYSBP', 'CIGPDAY', 'BMI', 'BPMEDS', 'HEARTRTE',
       'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP', 'TIME', 'HDLC', 'LDLC',
       'DEATH', 'CVD', 'HYPERTEN', 'TIMEMI', 'TIMEMIFC', 'TIMESTRK',
       'TIMEHYP']

iii. Both:

Best hyperparameters:  {'alpha': 0.01, 'l1_ratio': 0.1}
Best performance:  0.8272130163036439
Mean squared error:  91.56966298272421
R2 score:  0.8244894789043987

no of features selected: 22
selected features: 

       ['RANDID', 'SEX', 'AGE', 'SYSBP', 'CIGPDAY', 'BMI', 'BPMEDS', 'HEARTRTE',
       'GLUCOSE', 'educ', 'PREVCHD', 'PREVHYP', 'TIME', 'HDLC', 'LDLC',
       'DEATH', 'CVD', 'HYPERTEN', 'TIMEMI', 'TIMEMIFC', 'TIMESTRK',
       'TIMEHYP']

## GWAS using other methods:

### SVM + RFE

Recursive Feature Elimination (RFE) is a feature selection method that works by recursively removing attributes and building a model on those attributes that remain. It is a wrapper method, which means that it uses a model to identify the subset of features to keep. 
Here are the steps:

•	Choose a predictive model, such as support vector machine (SVM), and a scoring metric, such as mean squared error or accuracy.
•	Fit the model to the entire dataset and obtain the importance scores of all features.
•	Remove the least important feature(s) and fit the model again on the remaining features.
•	Repeat steps 2 and 3 until a desired number of features is reached.

The recursive feature elimination process is typically evaluated using cross-validation, in which the model is trained and tested on different subsets of the data to prevent overfitting. 
At each iteration, the cross-validation score is recorded and the number of features is also recorded. The optimal number of features is often determined by finding the number of features that results in the highest cross-validation score.

### Analysis

R-squared score: 0.643
Mean Squared Error: 186.372
Number of features selected:  20
Selected features: 

        ['SEX', 'TOTCHOL', 'AGE', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BPMEDS',
       'GLUCOSE', 'PREVCHD', 'PREVHYP', 'LDLC', 'ANGINA', 'ANYCHD', 'HYPERTEN',
       'TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMEDTH', 'TIMEHYP']








