# Drag prediction over rough surfaces via Machine Learning

## Project motivation💡 
This project seeks to provide insights of leveraging various ML models to develop accurate estimation of roughness-induced drag in many industrial applications. It is of great economic value as it significantly affects the energy expenditure and carbon emission budge. The source of roughness, depending on industrial process and application, can arise from biofouling in shipping, chemical process causing erosion in reactor/bed, ice accretion wearing down the leading edge of airfoil and turbine blade, sand scouring around the offshore wind turbine foundation, and simply surface coating. An accurate model for drag prediction is important to support the decision of energy expenditure and carbon emission. 
## New approach -- Machine learning 🕸️
Traditionally, measuring drag heavily depends on resourse-intensive CFD, towing tank and lab-scale experiments (wind tunnel, water channel), despite of extensive emprical correlation and physical models developed. Given the enormous amount of surface topographies, it has been challenging to both characterize roughness and run experiments for many of roughness. Therefore, we are trying to leverage machine learning to develop different predictive models to find both reliable and cost-effective model.  
## Generating roughness dataset 💎
First, we numerically generated the irregular homogeneous rough surfaces (better if you have access to actual engineering surface with data augmentation to improve sampling). See 
I conducted the high-performance direct numerical simulations in a channel to obtain the training and testing data, and also developed supervised learning models including support vector machine (SVR), multi-layer perceptron (MLP) and convolutional neural network (CNN). . Based on common characteristics of realistic surfaces, I numerically generated the database of rough surfaces with various topographical catagories.
## Goal -- correct mapping from surface topography to drag 📽️
The focus is to build up a non-linear mapping from surface topographical parameters to velocity deficit $\Delta U^+$ (a scalar) as a drag measure  . 
First, we numerically generated a dataset of ~1000 homogeneous roughness catagorized by Gaussianity and isotropy. We conducted the GPU-accelerated DNSs in a channel flow over those surfaces on cluster 'Tetralith' to obtain high-quality training and testing data. 
We adopted supervised learning (LR, SVM, MLP, CNN) to learn the non-linear mapping. SVM slightly outperforms more exhausting neural networks by balancing both accuracy and computational cost.

  
The codes in this repository feature a Python/Keras implementation of three predictive models, including SVR, multi-layer perceptron and convolutional neural network, to 
predict the momentum deficit over the homogeneous and irregular rough surfaces. The surfaces are numerically generated (./src/surf_generator) and consist
of five types. The labeled data are generated using [CaNS](https://github.com/CaNS-World/CaNS). 

<div align="center">
  
## Prediction Workflow 
  <img src="https://github.com/user-attachments/assets/36aaee46-e288-4e4c-80f3-5885e3141946" width="500" />

</div>

<!-- TOC -->
[🔌 predictive ML models](codes) <br />
[🚦 height map data](#data) <br />



<!-- TOC -->



