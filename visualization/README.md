### Visualize data relationship for different roughness type
<!---title--->
Get a glimpse of the relationship between input and output
* Note that some features (e.g. skewness, flatness) are *cluster* data;
* A clear linear relationship between (e.g. $k_{rms}, ES_x$) and $\Delta U^+$
<div align="center">
  <img src="https://github.com/user-attachments/assets/bac0b1d1-d020-4c2e-88f6-4c498d729b96" width=800 \>
</div>

<!---title--->
Pearson coefficient map [R](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) to show linear relationship of inter-parameters and parameter-drag
* The relationship between the parameters is also the source of non-linearity in this regression problem;
* The input data is normalized by min-max or standard deviation
* One can see that the non-linear interaction varies with the roughness type, t.ex. the valley-dominant roughness presents highly non-linearity
<div align="center">
  <img src="https://github.com/user-attachments/assets/2f7a8088-2155-4b4c-bec3-9a6861649808" width="1000" />
</div>
  
