# Treeo.AI: A masks to diameter service based on deeplabV3+

A Script to calculate the diameter of a trunk from the output of deeplabV3+, trained with treeo dataset.

deeplabV3+ trained on the treeo dataset performs a pixelwise classification on an image.
Possible classes are:
- trunk
- card
- background


Input: 
- predictions from deepLabV3+ : one dimensional predictions Array with length 224*224 = 50176

Outputs: 
- 0, if no card or no trunk detected or if predictions has not the size of 50176
- Diameter in cm, if card and trunk are detected and if predictions has size of 50176

Requirements:
- OpenCV, Can be downloaded from here: https://docs.opencv.org/master/opencv.js

## Example and Visualization:

Using the model trained with [deeplab-custom-dataset](https://github.com/Johannes0Horn/deeplab-custom-dataset "deeplab-custom-dataset") one can predict the diameter of a trunk from a photo of the trunk and a regular credit card.

[Example in Browser](https://johannes0horn.github.io/Treeo.AI/ "Example")

![alt text](https://github.com/Johannes0Horn/masksToDiameterService/blob/master/screenshot.png)

