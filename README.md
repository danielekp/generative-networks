# generative-networks 

## PixelCNN
The pixelcnn.py file is an implementaion of conditional generative model called 'pixelCNN'. The model generates the images one pixel per time, using a stack of masked convolutional layers. To condition the generation, the input of each masked layer is added to a 10-dimensional (10 digits) one-hot coded vector of the desidered class multiplied with parameters of a fully-connected layer.

## DCGAN
The dcgan.py file is an implementation of deep convolutional GAN model. The data is generated by applying a nonlinear transformation to samples drawn from the standard normal distribution. 
