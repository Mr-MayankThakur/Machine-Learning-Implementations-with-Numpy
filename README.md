# Machine Learning Implementations with Numpy
![GitHub repo size](https://img.shields.io/github/repo-size/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy)
[![made-with-latex](https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg)](https://www.latex-project.org/)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


It is crucial to how we could implement the machine learning formulas and methods in computers. Matrices are the special tools perfect for this task. This project contains basic implementations of machine learning models using numpy and scipy's minimize function. 

Project structure
-----------------

    ├─ models      <- The folder containing basic functions implemented in numpy for the project.
    │ │
    │ ├─ data_preprocessing.py   <- The functions used to prepare/modify the training data.
    │ ├─ linear_regression.py    <- The functions used to train and analyse the linear regression model.
    │ ├─ logistic_regression.py  <- The functions used to train and analyse the logistic regression model.
    │ └─ neural_network.py       <- The functions used to train and analyse the neural network model.
    │
    ├─ Linear Regression    <- The folder containing jupyter notebooks and scripts for linear regression.
    │
    ├─ Logistic Regression  <- The folder containing jupyter notebooks and scripts for logistic regression.
    │
    ├─ Neural Networks      <- The folder containing jupyter notebooks and scripts for neural networks.
    │
    ├─ environment.yml   <- The anaconda environment file for the project
    └─ README.md         <- The readme file for the project, explaining the basics to new developers

Getting started
---------------

### Installation / Usage

The .ipynb files, which are the core of this repository, are interactive Jupyter Notebooks. You can directly use a static, rendered version of the notebook by clicking on it. GitHub has an implemented notebook-viewer.
Further you can inspect notebooks on NBviewer by following the links above.

However, for working interactively with the notebooks (recommended), you either have to install Python + Jupyter (e.g. by using Anaconda), clone the repository and start a server. Or start them in an online, interactive environment, such as Binder (see below).
### Method 2: Offline
Execute these steps from an Anaconda prompt to get started with this project::

    conda env create -f environment.yml -n machine_learning_implementations_with_numpy
    conda activate machine_learning_implementations_with_numpy

Example Notebooks
-----------------
<a href="Linear_Regression\Univariate Linear Regression.ipynb">Univariate Linear Regression</a>
[![](https://img.shields.io/badge/open%20with-nbviewer-green)](https://nbviewer.jupyter.org/github/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/blob/master/Linear_Regression/Univariate%20Linear%20Regression.ipynb)

<a href="Linear_Regression\Multi-Variable Linear Regression.ipynb">Multi-Variable Linear Regression</a>
[![](https://img.shields.io/badge/open%20with-nbviewer-green)](https://nbviewer.jupyter.org/github/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/blob/master/Linear_Regression/Multi-Variable%20Linear%20Regression.ipynb)

<a href="Logistic_Regression\Logistic Regression (without regularization).ipynb">Logistic Regression (without regularization)</a>
[![](https://img.shields.io/badge/open%20with-nbviewer-green)](https://nbviewer.jupyter.org/github/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/blob/master/Logistic_Regression/Logistic%20Regression%20%28without%20Regularization%29.ipynb)

<a href="Logistic_Regression\Logistic Regression (with regularization).ipynb">Logistic Regression (with regularization)</a>
[![](https://img.shields.io/badge/open%20with-nbviewer-green)](https://nbviewer.jupyter.org/github/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/blob/master/Logistic_Regression/Logistic%20Regression%20%28with%20regularization%29.ipynb)

<a href="Logistic_Regression\Multi-Class Classification.ipynb">Multi-Class Classification</a>
[![](https://img.shields.io/badge/open%20with-nbviewer-green)](https://nbviewer.jupyter.org/github/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/blob/master/Logistic_Regression/Multi-Class%20Classification.ipynb)

<a href="Model Performance Evaluation/Regularized Linear Regression and Bias vs Variance.ipynb">Regularized Linear Regression and Bias vs Variance</a>
[![](https://img.shields.io/badge/open%20with-nbviewer-green)](https://nbviewer.jupyter.org/github/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/blob/master/Model%20Performance%20Evaluation/Regularized%20Linear%20Regression%20and%20Bias%20vs%20Variance.ipynb)

<a href="Neural Networks/neural_network_feed_forwarding.ipynb">Neural Network Feed Forwarding</a>
[![](https://img.shields.io/badge/open%20with-nbviewer-green)](https://nbviewer.jupyter.org/github/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/blob/master/Neural%20Networks/neural_network_feed_forwarding.ipynb)

<a href="Neural Networks/neural_network_complete.ipynb">Neural Network Complete</a>
[![](https://img.shields.io/badge/open%20with-nbviewer-green)](https://nbviewer.jupyter.org/github/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/blob/master/Neural%20Networks/neural_network_complete.ipynb)


Dependency
==========

  * numpy
  * matplotlib
  * scipy
  * seaborn

Tested on Ubuntu 18.04 LTS.

## Maintainers
[@MayankThakur](https://github.com/Mr-MayankThakur).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/Mr-MayankThakur/Machine-Learning-Implementations-with-Numpy/issues/new/choose) or submit PRs.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
