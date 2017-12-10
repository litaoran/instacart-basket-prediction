# Instacart Market Basket Analysis

This is the project for kaggle contest: Instacart Market Basket Analysis.

#How to run
To run the project, just use sh run.sh to run a sequence of python code which includes data preprocessing, feature extraction and model training, the submit.py will return the final results for submission.

#Requirement
    hardware resource:
        aws g3.4xlarge :
        GPU: 8 GB
        Memory:	122 GB
        Disk: > 50 GB

    libraries:
        lightgbm==2.0.11
        numpy==1.13.3
        pandas==0.21.0
        scikit-learn==0.19.1
        tensorflow==1.4.0

#Reference
we reference https://www.kaggle.com/mmueller/f1-score-expectation-maximization-in-o-n to add F1 optimizer. And we also reference some code from sjvasquez's solution(https://github.com/sjvasquez/instacart-basket-prediction).
