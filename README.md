# Predicting shear-induced particle migration in channel flow with Physics-Informed Neural Networks

This repository contains PyTorch implementations of physics-informed neural networks (PINNs), which learn the known velocity profile of fluid through a small channel, and use this along with enforced physical laws to approximate the unknown particle-volume-fraction profile of particle suspensions. Later models, such as pinn_sbm_lift_force_7.py, include a lift force term to describe the deformable body behavior of red blood cells, resulting in greater concentrations towards the center flow, and values of zero for the concentration near the walls. 

## Results

See the image below for the results for a channel width is 25 micrometers, beta (lift force parameter) is 1.1, and bulk concentration is 0.2. Notice the zero values for the particle-volume-fraction near the walls. The model learned this entirely by enforcing the governing equations and boundary conditions.
![PINN Solution](assets/Channel_width_25um_beta_1_1_phi_0_2.png)
See the image below for the results for a channel width is 50 micrometers, beta (lift force parameter) is 1.2, and bulk concentration is 0.1. Notice the large zero regions near the walls, which are correctly predicted by the model. 
![PINN Solution](assets/Channel_width_50um_beta_1_2_phi_0_1.png)
The following three images are of models which do not include a lift force for modeling blood flow
![PINN Solution](assets/bulk_03.png)
![PINN Solution](assets/bulk_04.png)
![PINN Solution](assets/bulk_05.png)

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

@article{McClenny2023SAPINN,
  author  = {Levi D. McClenny and Ulisses M. Braga-Neto},
  title   = {Self-adaptive physics-informed neural networks},
  journal = {Journal of Computational Physics},
  volume  = {474},
  pages   = {111722},
  year    = {2023},
  doi     = {10.1016/j.jcp.2022.111722}
}

@article{Tancik2020Fourier,
  author  = {Matthew Tancik and Pratul P. Srinivasan and Ben Mildenhall and 
             Sara Fridovich-Keil and Nithin Raghavan and Utkarsh Singhal and 
             Ravi Ramamoorthi and Jonathan T. Barron and Ren Ng},
  title   = {Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains},
  journal = {arXiv preprint arXiv:2006.10739},
  year    = {2020},
  doi     = {10.48550/arXiv.2006.10739},
  url     = {https://arxiv.org/abs/2006.10739}
