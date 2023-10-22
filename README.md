# Contrastive Deep Encoding Enables Uncertainty-aware Machine-learning-assisted Histopathology

This is the repository of the paper [Contrastive Deep Encoding Enables Uncertainty-aware Machine-learning-assisted Histopathology](https://arxiv.org/abs/2309.07113).


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

## Dataset preparation

We used publicly available datasets in our work.

* NCT-CRC-HE-100K https://zenodo.org/records/1214456
* Patch Camelyon https://patchcamelyon.grand-challenge.org/

The dataset should follow the following structure

```
-- NCT-CRC-HE-100K
  -- ADI
      -- <image-name>.jpg
  -- BACK
      -- <image-name>.jpg

-- CRC-HE-VAL-7K
  -- ADI
      -- <image-name>.jpg
  -- BACK
      -- <image-name>.jpg
```
The dataloader reads the data though csv files. Example csv files have been provided in the google [drive](https://drive.google.com/drive/folders/1VepRvPOZ_B6CnH9kWzB0CBV60pB7BhiL?usp=share_link) for Patch camelyon and NCT-CRC-HE-100K for 100% labels available scenario.

## Train the model

To pretrain the model 

```
python main.py --training_data_csv <path-to-csv-file> --validation_data_csv <path-to-csv-file> --test_data_csv <path-to-csv-file> --dataset "nct100k" --data_input_dir <data-input-directory> --save_dir <save-directory>
```

To finetune the model 

```
python linear.py --model_path <pretrained_model_path> --training_data_csv <path-to-csv-file> --validation_data_csv <path-to-csv-file> --test_data_csv <path-to-csv-file> --dataset "nct100k" --data_input_dir <data-input-directory> --save_dir <save-directory> --finetune --uncertainty True
```


For knowledge distillation

```
python distillation.py --model_path <finetuned_model_path> --training_data_csv <path-to-csv-file> --validation_data_csv <path-to-csv-file> --test_data_csv <path-to-csv-file> --dataset "nct100k" --data_input_dir <data-input-directory> --save_dir <save-directory> --finetune --uncertainty True
```

The returned output for NCT-CRC-HE-100K would be similar to the output file available in the drive [link](https://drive.google.com/drive/folders/17uqMTyLAC6oJ26p6lEjV38DAsfTEJZ86?usp=share_link).

## Hardware resources

The results in the paper were produced using 4 Nvidia A100 GPUs with distributed training. `Batch_size` was set to 128. With this setting, the time it took for NCT-CRC-HE-100k data to pretrain for 500 epochs were 1.5 days, to finetune the model 12 hours and for knowledge distillation it took 12 hours.

## TSNE Visualizations

We have provided the visualization tool we used to analyse the trend of the model predictions with uncertainty score in the [drive](https://drive.google.com/file/d/1QqGi_ORw7tsbnN9mNQMLBYAdIXn__LGZ/view?usp=share_link) . The colour coding used here is 

* 0 (Green)  - Correct predictions with low uncertainty scores
* 1 (Red)    - Correct predictions with high uncertainty scores
* 2 (Blue)   - Incorrect predictions with high uncertainty scores
* 3 (Yellow) - Incorrect predictions with low uncertainty scores

## Acknowledgements

The code has parts extracted from

* https://github.com/k-stacke/ssl-pathology
* https://github.com/dougbrion/pytorch-classification-uncertainty



