[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/gbohner/ucl_introductiontomachinelearning_dimensionalityreduction)

# Introduction To Machine Learning - Dimensionality Reduction
Tutorial material for UCL's Introduction To Machine Learning course

The repository contains python code organised as:
* An ```index.ipynb``` notebook, that guides you through the tasks and makes visualisation easy.
* A ```dimred_funcs.py``` file that gathers the supporting functions as well as some solutions if you are lost.

There are two ways to use this repository:
* Firstly, you could click on the button at the top to launch an interactive coding environment in your browser that runs on our servers. This might be ok for the class now, but it is a bit slow and has limited customisation options if you want to use python in the future.
* Or you could download and run all the code on your own machine. Sadly python does not support Windows systems very well, so the installation tutorial below will mostly apply to Unix-based systems (Linux and Mac OSX).

## Using python locally

1. First install the **2.7 version** of python as described at https://docs.continuum.io/anaconda/install . This contains numerous packages that will be useful for us (mainly *jupyter*, *numpy*, *scipy* and *matplotlib*).
2. Download the code from this repository (either as zip, or using the ```git``` protocol)
3. Launch a ```terminal``` and go to the main directory of the repository.
3.1 Install the machine-learning package we'll require from python by typing ```conda install scikit-learn```
3.2 Launch a notebook server locally that will let you code via your browser by ```jupyter notebook```. This will open a new tab in your favourite browser from which you can open the individual files and run the code. 


## Using the notebook
You can run the code in individual blocks (```ctrl+enter```) and examine the outcome right below the code. Many of the plots in the notebook is interactive.
The course material (explaining the mathematics behind these algorithms) can be found at http://www.gatsby.ucl.ac.uk/~maneesh/dimred/
