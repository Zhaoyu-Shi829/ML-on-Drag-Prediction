## Directory structure: </br>
├── src/             </br>
│   ├── model.py               # Model architecture and BO configuration            </br>
│   ├── train.py               # Training, validation, and testing logic            </br>
│   ├── data_processing.py     # Data loading and preprocessing functions           </br>
│   ├── config.py              # Configuration parameters                           </br>
│   └── utils.py               # Utility functions like normalization and plotting  </br>
├── README.md                  # Project description                                </br>
├── requirements.txt           # List of dependencies                               </br>
└── main.py                    # Script to start training or testing                </br>



## Model configuration
* Model inputs: 
  * LR, SVR, MLP use topographical parameters of surface height: `height features` ($k_{rms}, k_{avg}, k_c$) and `effective slopes` ($Ex_x, Ex_z$) 
  * CNN use height map as image input $k(n_x,n_z)$, here is an illustration of a Gaussian surface.
 
    <div align="left">
      <img src="https://github.com/user-attachments/assets/a0b7cc35-6e2b-4518-acd9-c67a4fca6584" width=400 />
    </div>
    
## Bayesian hyperparams optimization (BO): 
* Given the current training data is relarively small size compared to computer vision applications, it is feasible to apply BO to search the optimals for MLP and CNN;
* In this practice using [Scikit Optimize](https://github.com/user-attachments/assets/a0b7cc35-6e2b-4518-acd9-c67a4fca6584), the `search space` for includes the number of layers/neurons, the number of blocks/filters, the kernel size and batch size, the learning rate and L2 regularization paramter $\lambda_2$. Here is the BO process sketch:
  
    <div align="left">
      <img src="https://github.com/user-attachments/assets/dc81c5fa-f1ed-432c-a74d-260f0332dadd" width=400 />
    </div>
  

