# MFI - Measure of Feature Importance. 

`MFI` is an algorithm that was developed to explain arbitrary classifiers in two ways:

1. model-based explanation: what has the classifier learned in total?

2. instance-based explanation: Given a specific data point (instance), which are the important features of this data point that drive the classifier prediction?

# Installation
Download or clone:

`https://github.com/mcvidomi/MFI.git`

# Demo

run `demo.py`. It will take about 1 min.

USPS (United States Postal Service - handwritten digits) dataset will be downloaded in data/usps.

An SVM with an RBF kernel is trained on the data. Afterwards `MFI` is computed for the instance-based explanation (exemplarly 4 digits were chosen) and for the model-based explanation.

# Results

## Instance-based results
The first row shows the raw digits.
The second row shows the shape of the digits over the `MFI` results, respectively. On the top of the image the prediction score of the classifier is plotted - <0: decide for '3' and >0: decide for '8'.
![alt tag](/mfi_ibr.png)

## Model-based results

The result heat map shows the important pixel for the classifier to decide for a '3' instead of an '8'. 
![alt tag](/mfi_mbr.png)


