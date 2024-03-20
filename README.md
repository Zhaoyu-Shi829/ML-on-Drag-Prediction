# Drag prediction over irregular rough surfaces via Machine Learning

## Introduction
The codes in this repository feature a Python/Keras implementation of three predictive models, including SVR, multi-layer perceptron and convolutional neural network, to 
predict the momentum deficit $\Delta U^+$ (a scalar) over the homogeneous and irregular rough surfaces. The surfaces are numerically generated (./src/surf_generator) and consist
of five types. The labeled data are generated using CaNS [https://github.com/CaNS-World/CaNS]. 
