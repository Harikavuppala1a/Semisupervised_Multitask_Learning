This repository contains code for our paper titled "Multi-label Fine-grained Sexism Classification using Semi-supervised Multi-task Learning" published in COLING-2020.

Abstract: Sexism, a pervasive form of oppression, causes profound suffering through various manifestations. Given the rising number of experiences of sexism reported online, categorizing these recollections automatically can aid the fight against sexism, as it can facilitate effective analyses by gender studies researchers and government officials involved in policy making. In this paper, we explore the fine-grained, multi-label classification of accounts (reports) of sexism. To the best of our knowledge, we consider substantially more categories of sexism than any published work through our 23-class problem formulation. Moreover, we propose a multi-task approach for fine-grained multi-label sexism classification that leverages several supporting tasks without incurring any manual labeling cost. Unlabeled accounts of sexism are utilized through unsupervised learning to help construct our multi-task setup. We also devise objective functions that exploit label correlations in the training data explicitly. Multiple proposed methods outperform the state-of-the-art for multi-label sexism classification on a recently released dataset across five standard metrics

Sample structure to run the code: Python main.py data/config_deeplearning.txt. 

If you use this code for any research, please cite this paper