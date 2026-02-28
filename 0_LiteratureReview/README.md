# Literature Review

Approaches or solutions that have been tried before and similar projects.

**Summary of Each Work**:



- **Source 1**: [Physics-Informed Neural Networks for Advanced Thermal Management in Electronics and Battery Systems: A Review of Recent Developments and Future Prospects]

  - **[Link](https://www.mdpi.com/2313-0105/11/6/204)**
  - **Objective**: The paper examines recent advancements in PINN architectures, comparing them to traditional Computational Fluid Dynamics (CFD) methods. It explores the integration of physical laws—such as energy conservation—directly into the loss function, as well as hybrid models combining PINNs with Long Short-Term Memory (LSTM) networks for transient temperature prediction.
  - **Methods**:The paper examines recent advancements in PINN architectures, comparing them to traditional Computational Fluid Dynamics (CFD) methods. It explores the integration of physical laws—such as energy conservation—directly into the loss function, as well as hybrid models combining PINNs with Long Short-Term Memory (LSTM) networks for transient temperature prediction.
  - **Outcomes**: PINNs were found to be significantly faster than traditional solvers, in some cases up to 300,000 times, while maintaining high accuracy with temperature differences of less than 0.1 K. In battery management, the hybrid PINN-LSTM approach achieved high predictive fidelity with a mean absolute error (MAE) as low as 0.2875 °C.
  - **Relation to the Project**: This source provides the justification for using PINNs in thermal problems, demonstrating that they can handle the complex, real-time demands of heat diffusion in physical systems with much higher efficiency than classical numerical solvers.

- **Source 2**: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains]

  - **[Link]**: [https://arxiv.org/pdf/2006.10739](https://arxiv.org/pdf/2006.10739)
  - **Objective**: To overcome the "spectral bias" of coordinate-based MLPs, which causes neural networks to prioritize low-frequency components and struggle with high-frequency details.
  - **Methods**: Mapping low-dimensional input coordinates ($x, y, t$) into a higher-dimensional feature space using a Fourier feature mapping (Sinusoidal transforms) before passing them to the MLP.
  - **Outcomes**: Proved that this mapping allows the user to tune the bandwidth of the Neural Tangent Kernel (NTK), enabling the network to learn high-frequency spatial and temporal data much faster.
  - **Relation to the Project**: Directly justifies the use of the **Sinusoidal Encoder** in the `Thermal_PINN` architecture. This specific component is what allows the model to resolve the sharp 170°C peak of the localized heat source.


- **Source 3**: [SIMULATION OF 2-DIMENSIONAL HEAT CONDUCTION WITH PHYSICS-INFORMED NEURAL NETWORKS]

  - **[Link](https://forschung.rwu.de/sites/forschung/files/2024-11/Bachelor_Thesis_Samman_Aryal_Final.pdf)**
  - **Objective**: To develop and evaluate a PINN-based framework for simulating 2D heat conduction under various boundary conditions and heat sources.
  - **Methods**: The thesis implements a PINN framework using PyTorch that incorporates the laws of heat conduction directly into the network design. The method is tested through case studies, including localized hot regions and square plates, and its performance is benchmarked against traditional Finite Element Method (FEM) and Finite Difference Method (FDM) techniques.
  - **Outcomes**: The research confirms that PINNs accurately capture heat distribution and are a robust alternative to traditional numerical techniques. The findings emphasize the framework's flexibility and potential for increased computational efficiency in complex thermal systems.
  - **Relation to the Project**: This source offers a comprehensive academic validation of the PINN approach for 2D heat conduction. It addresses specific challenges like irregular heat sources and varying boundary conditions, which are central to modeling 2D heat diffusion accurately.
