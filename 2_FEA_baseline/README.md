# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline  Type:** FEA in fenics for 2D heat diffusion
- **Rationale:** Finite element analysis is the gold standard for performing studies on PDEs, it is proven to be accurate but slow.

### Model Performance
- **Evaluation Metric:** Execution time
- **Performance Score:**  460 s


### Evaluation Methodology
- **FEA Setup:** the mesh is setup as an 80x f grid, with a dt of 1ms, and it simulates up to 0.3 seconds, fot the PDE it uses Implicit euler version of the heat equation 
  $$
  (1 - \Delta t \alpha \nabla^2) u^{n+1} = u^n + \Delta t f^{n+1}
  $$

- **Evaluation Metrics:** Since FEA is the closest we can get to ground truth for a heat diffusion problem without actual physical experimentation, its accuracy is out of the scope of this work, instead we focus on the exception time as it is its main disadvantage 

### Metric Practical Relevance
In real world design processes, there exist a lot of iteration, therefore the execution time of the heat diffusion analysis hinders the design speed significantly, byt being able to improve the evaluation time, even at the cost of some accuracy faster iterations could e accomplished and later on validate a final sample with a FEA method

## Next Steps
This baseline model serves as a reference point for evaluating the accuracy of the PINN.
