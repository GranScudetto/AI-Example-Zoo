# CIFAR10 - Classification
## about CIFAR
The two CIFAR datasets (CIFAR10 and CIFAR100) are public available datasets and nowadays they are integrated into most Machine Learning frameworks. Basically they represent a collection of tiny images and differ in the amount of classes to distinguish between. In Image Classification they are widely used for CNN based Classifier approaches. The abbreviation is based on the **C**anadian **I**nstitute **f**or **A**dvanced **R**esearch who published the dataset as part of their tiny image dataset in 2009. If you want to dive deeper into their work I recommend the corresponding publication: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

The Data consists of **60.000** images in total, from which **50.000** are used for training and **10.000** are reserved for testing.
Each image has a *width* and a *height* of **32 pixels** and shows one of the following **10 classes**:

['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

A overview of the data and its quality can be obtained in the samples below:
![Cifar10 Examples](https://raw.githubusercontent.com/wiki/GranScudetto/AI-Example-Zoo/images/datasets/cifar10/samples.png)

## next implementation goals
- [ ] interactive visualization for results
- [ ] load saved model
- [ ] fix experiment folder location...
- [ ] model performance!
- [ ] data augmentation
- [ ] data preprocessing (mean, variance, ...)
- [ ] documentation
