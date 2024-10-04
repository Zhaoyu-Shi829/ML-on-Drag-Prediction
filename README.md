# Drag prediction over rough surfaces via Machine Learning

## Project motivation💡 
This project seeks to provide insights of leveraging various ML models to develop accurate estimation of roughness-induced drag in many industrial applications. It is of great economic value as it significantly affects the energy expenditure and carbon emission budge. The source of roughness, depending on industrial process and application, can arise from biofouling in shipping, chemical process causing erosion in reactor/bed, ice accretion wearing down the leading edge of airfoil and turbine blade, sand scouring around the offshore wind turbine foundation, and simply surface coating. An accurate model for drag prediction is important to support the decision of energy expenditure and carbon emission. 

## New approach -- Machine learning 🕸️
Traditionally, measuring drag heavily depends on resourse-intensive CFD, towing tank and lab-scale experiments (wind tunnel, water channel), despite of extensive emprical correlation and physical models developed. Given the enormous amount of surface topographies, it has been challenging to both characterize roughness and run experiments for many of roughness. Therefore, we are trying to leverage machine learning to develop different predictive models to find both reliable and cost-effective model. Supervised learning is used to develop the predictive models, including linear regression, support vector machine (SVR), multi-layer perceptron (MLP) and convolutional neural network (CNN). 
* [🔗 predictive ML models](model)
  
## Generating roughness dataset 💎
The dataset of ~1000 irregular homogeneous rough surfaces are numerically generated catagorized by Gaussianity and isotropy (better if you have access to actual engineering surface with data augmentation to improve sampling). The direct numerical simulations (DNSs) of a mini channel over the surfaces were conducted on HPC cluster 'Tetralith' to obtain the training and testing data using the GPU-accelerated solver [CaNS](https://github.com/CaNS-World/CaNS). 
* [🔗 height map data](data): original height map (structured data corresponding to mesh resolution $$[n_x,n_z]$$) 

### Goal -- correct mapping from surface topography to drag 📽️
The focus is to build up a non-linear mapping from surface topographical parameters to velocity deficit $\Delta U^+$ (a scalar) as a drag measure. 

<div align="center">
  
## Prediction Workflow 
  <img src="https://github.com/user-attachments/assets/36aaee46-e288-4e4c-80f3-5885e3141946" width="500" />

</div>

To learn more about our general approach, read our papers: <br />
*  [Drag prediction of rough-wall turbulent flow using data-driven regression](http://arxiv.org/abs/2405.09256) <br />
*  [Data-driven discovery of drag-inducing elements on a rough surface through convolutional neural networks}](https://doi.org/10.1063/5.0223064)






