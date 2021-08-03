This repository contains code for our paper titled "Multi-label Fine-grained Sexism Classification using Semi-supervised Multi-task Learning" published in COLING-2020.

Abstract: Sexism, a form of oppression based on one's sex, manifests itself in numerous ways and causes enormous suffering. In view of the growing number of experiences of sexism reported online, categorizing these recollections automatically can assist the fight against sexism, as it can facilitate effective analyses by gender studies researchers and government officials involved in policy making. In this paper, we investigate the fine-grained, multi-label classification of accounts (reports) of sexism. To the best of our knowledge, we work with considerably more categories of sexism than any published work through our 23-class problem formulation. Moreover, we propose a multi-task approach for fine-grained multi-label sexism classification that leverages several supporting tasks without incurring any manual labeling cost. Unlabeled accounts of sexism are utilized through unsupervised learning to help construct our multi-task setup. We also devise objective functions that exploit label correlations in the training data explicitly. Multiple proposed methods outperform the state-of-the-art for multi-label sexism classification on a recently released dataset across five standard metrics.

One can run the code using: python main.py data/config_deeplearning.txt. 

If you use this code for any research, please cite our COLING paper.

@inproceedings{abburi2020semi,
  title={Semi-supervised Multi-task Learning for Multi-label Fine-grained Sexism Classification},
  author={Abburi, Harika and Parikh, Pulkit and Chhaya, Niyati and Varma, Vasudeva},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={5810--5820},
  year={2020}
}
