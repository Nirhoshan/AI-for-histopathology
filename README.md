# Contrastive Deep Encoding Enables Uncertainty-aware Machine-learning-assisted Histopathology

This is the repository of the paper [Contrastive Deep Encoding Enables Uncertainty-aware Machine-learning-assisted Histopathology](https://arxiv.org/abs/2310.04429).


## Brief overview of the paper

Deep neural network models can learn clinically relevant features from millions of histopathology images. However generating high-quality annotations to train such models for each hospital, each cancer type, and each diagnostic task is prohibitively laborious. On the other hand, terabytes of training data -- while lacking reliable annotations -- are readily available in the public domain in some cases. In this work, we explore how these large datasets can be consciously utilized to pre-train deep networks to encode informative representations. We then fine-tune our pre-trained models on a fraction of annotated training data to perform specific downstream tasks. We show that our approach can reach the state-of-the-art (SOTA) for patch-level classification with only 1-10% randomly selected annotations compared to other SOTA approaches. Moreover, we propose an uncertainty-aware loss function, to quantify the model confidence during inference. Quantified uncertainty helps experts select the best instances to label for further training. Our uncertainty-aware labeling reaches the SOTA with significantly fewer annotations compared to random labeling. Last, we demonstrate how our pre-trained encoders can surpass current SOTA for whole-slide image classification with weak supervision. Our work lays the foundation for data and task-agnostic pre-trained deep networks with quantified uncertainty.

## Package Requirements
We used Python 3.9 version.
Following packages are required.

* Pytorch				
*	Torchvision			
*	numpy
*	pandas
*	matplotlib
*	pillow
*	tqdm
*	lmdb
*	albumentations

