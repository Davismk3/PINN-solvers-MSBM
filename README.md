# Predicting shear-induced particle migration in channel flow with Physics-Informed Neural Networks

This repository contains PyTorch implementations of physics-informed neural networks (PINNs), which learn the known velocity profile of fluid through a small channel, and use this along with enforced physical laws to approximate the unknown particle-volume-fraction profile of particle suspensions. 

The script **`pinn_sbm.py`** is currently set to reproduce **Case 1** of the SBM study by Dbouk *et al.* (2013).

## Results

![PINN Solution](assets/3.png)
![PINN Solution](assets/4.png)
![PINN Solution](assets/5.png)

## Reference
@article{Dbouk2013SBM,
  author  = {Dbouk, Talib and Lemaire, Elisabeth and Lobry, Laurent and Moukalled, Fady},
  title   = {Shear-induced particle migration: Predictions from experimental evaluation of the particle stress tensor},
  journal = {Journal of Non-Newtonian Fluid Mechanics},
  volume  = {198},
  pages   = {78--95},
  year    = {2013},
  doi     = {10.1016/j.jnnfm.2013.03.006}
}
