# Code for the Kaggle PANDA Prostate Cancer Grading Competition

Ultimately, during this competition, the best model I've found is to use the ResNext50 pretrained model with some data augmentation. This yielded a final leaderboard score of approximately 0.90.

The code for this is relatively short and in resnext50.py

I also tried using EfficientNet for this model but anything other than EfficientNet-B0 seems to overfit the data. The code for this can be found in efficientnet.py This yielded leaderboard scores between 0.8 and 0.9 typically

Since this competition required submissions to be done through Kaggle kernels, I had to ensure that the EfficientNet model code is identical on both sides. So the EfficientNet-PyTorch repository which is at the same git has as the Kaggle kernel is included in this repository. That code is not mine and I do not take credit for it.

Note visualise.py also provides some useful timing and visulaisation code for data loading. One problem with data loading here is that preprocessing the image into tiles seemed to use considerable CPU resources due to the large size of the images.

Finally, dataset.py and loader.py contain the utilities needed for loading the data.
