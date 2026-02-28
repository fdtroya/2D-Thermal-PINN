import numpy as np
import matplotlib.pyplot as plt

class Geometry():
    def __init__(self, regions, lx, ly, name='test', Q_loss=10, diff=None, rho_cp=None, backend='torch',T_MAX=1, T_MIN=0.0,X_MIN=-1, X_MAX=1, Y_MIN=-1, Y_MAX=1):
        self.backend = backend
        
        if self.backend == 'torch':
            import torch
            self.xp = torch
        elif self.backend == 'numpy':
            self.xp = np
        else:
            raise ValueError("Backend must be 'torch' or 'numpy'")

        self.regions = regions
        self.lx, self.ly = lx, ly
        self.Q_loss = Q_loss
        self.diff = diff
        self.rho_cp = rho_cp

        self.T_MAX = T_MAX
        self.T_MIN = T_MIN
        self.X_MIN = X_MIN
        self.X_MAX = X_MAX
        self.Y_MIN = Y_MIN
        self.Y_MAX = Y_MAX
        for name_key, props in self.regions.items():
            props["x_init_norm"] = (self.X_MAX - self.X_MIN) * props["x_init"] / self.lx + self.X_MIN
            props["x_fin_norm"]  = (self.X_MAX - self.X_MIN) * props["x_fin"] / self.lx + self.X_MIN
            props["y_init_norm"] = (self.Y_MAX - self.Y_MIN) * props["y_init"] / self.ly + self.Y_MIN
            props["y_fin_norm"]  = (self.Y_MAX - self.Y_MIN) * props["y_fin"] / self.ly + self.Y_MIN

        active = self.regions["active"]
        self.active_x_init = active["x_init"]
        self.active_y_init = active["y_init"]
        self.active_x_fin = active["x_fin"]
        self.active_y_fin = active["y_fin"]
        
        self.name = name
        self.init_localized_Q()


    
    def material_mask_regions(self, x, delta=2e-3, default_props=None):
        """
        x: [N,2] array or tensor strictly in [-1, 1].
        Returns: dict of {k, rho, cp} of shape [N,1]
        """
        N = x.shape[0]

        if default_props is None:
            default_props = {"k": 130.0, "rho": 2330.0, "cp": 700.0}

        if self.backend == 'torch':
            k = self.xp.full((N, 1), default_props["k"], device=x.device)
            rho = self.xp.full((N, 1), default_props["rho"], device=x.device)
            cp = self.xp.full((N, 1), default_props["cp"], device=x.device)
        else:
            k = self.xp.full((N, 1), default_props["k"])
            rho = self.xp.full((N, 1), default_props["rho"])
            cp = self.xp.full((N, 1), default_props["cp"])

    
        x_coord = x[:, 0]
        y_coord = x[:, 1]


        for props in self.regions.values():
            x_i = props["x_init_norm"]
            x_f = props["x_fin_norm"]
            y_i = props["y_init_norm"]
            y_f = props["y_fin_norm"]

            mask_x = 0.5 * (1 + self.xp.tanh((x_coord - x_i) / delta)) * \
                     0.5 * (1 + self.xp.tanh((x_f - x_coord) / delta))

            mask_y = 0.5 * (1 + self.xp.tanh((y_coord - y_i) / delta)) * \
                     0.5 * (1 + self.xp.tanh((y_f - y_coord) / delta))

            # Now this safely converts [N] -> [N, 1]
            mask = (mask_x * mask_y)[..., None] 

            k = k * (1 - mask) + props["k"] * mask
            rho = rho * (1 - mask) + props["rho"] * mask
            cp = cp * (1 - mask) + props["cp"] * mask

        return {"k": k, "rho": rho, "cp": cp}
    def init_localized_Q(self, region_name="active"):
        props = self.regions[region_name]
        
        # Now directly using the pre-calculated normalized boundaries
        self.xc_norm = (props["x_init_norm"] + props["x_fin_norm"]) / 2.0
        self.yc_norm = (props["y_init_norm"] + props["y_fin_norm"]) / 2.0
        
        self.w_norm = abs(props["x_fin_norm"] - props["x_init_norm"])
        self.h_norm = abs(props["y_fin_norm"] - props["y_init_norm"])

    def localized_Q_gaussian(self, x_norm, y_norm, t_norm, Q_max=1.125, softness=6.0, ramp_t0=3.0):
        rel_x = (x_norm - self.xc_norm) / (self.w_norm / 2.0)
        rel_y = (y_norm - self.yc_norm) / (self.h_norm / 2.0)
        
        # Reverted back to explicit .pow() for PyTorch autograd stability
        if self.backend == 'torch':
            mask_x = self.xp.exp(-self.xp.pow(rel_x**2, softness / 2.0))
            mask_y = self.xp.exp(-self.xp.pow(rel_y**2, softness / 2.0))
        else:
            mask_x = self.xp.exp(-((rel_x**2) ** (softness / 2.0)))
            mask_y = self.xp.exp(-((rel_y**2) ** (softness / 2.0)))
        
        spatial_mask = mask_x * mask_y
        
        if ramp_t0 == -1:
            time_factor = 1.0
        else:
            time_factor = self.xp.tanh(ramp_t0 * t_norm)**2
        
        return Q_max * spatial_mask * time_factor

    def material_mask_regions_optimized(self, x_coord, y_coord, delta=2e-3):
        k_bg = 130.0
        rhocp_bg = 2330.0 * 700.0
        
        k = self.xp.full_like(x_coord, k_bg)
        rho_cp = self.xp.full_like(x_coord, rhocp_bg)
        if self.backend == 'torch':
            rho_cp = rho_cp.detach()

        for props in self.regions.values():
            # Reverted to computing this dynamically to ensure exact graph 
            x_i = (self.X_MAX - self.X_MIN) * props["x_init"] / self.lx + self.X_MIN
            x_f = (self.X_MAX - self.X_MIN) * props["x_fin"] / self.lx + self.X_MIN
            y_i = (self.Y_MAX - self.Y_MIN) * props["y_init"] / self.ly + self.Y_MIN
            y_f = (self.Y_MAX - self.Y_MIN) * props["y_fin"] / self.ly + self.Y_MIN

            mask_x = 0.5 * (self.xp.tanh((x_coord - x_i) / delta) + self.xp.tanh((x_f - x_coord) / delta))
            mask_y = 0.5 * (self.xp.tanh((y_coord - y_i) / delta) + self.xp.tanh((y_f - y_coord) / delta))
            mask = mask_x * mask_y
            
            k = k + (props["k"] - k_bg) * mask
            
            rho_cp_val = props["rho"] * props["cp"]
            rho_cp = rho_cp + (rho_cp_val - rhocp_bg) * mask
            
            if self.backend == 'torch':
                rho_cp = rho_cp.detach()

        return k, rho_cp

    def material_mask_regions_step(self, x, default_props=None):
        """
        x: [N,2] array or tensor strictly in [-1, 1].
        Returns: dict of {k, rho, cp} of shape [N,1]
        """
        N = x.shape[0]

        if default_props is None:
            default_props = {"k": 130.0, "rho": 2330.0, "cp": 700.0}

        if self.backend == 'torch':
            k = self.xp.full((N, 1), default_props["k"], device=x.device)
            rho = self.xp.full((N, 1), default_props["rho"], device=x.device)
            cp = self.xp.full((N, 1), default_props["cp"], device=x.device)
        else:
            k = self.xp.full((N, 1), default_props["k"])
            rho = self.xp.full((N, 1), default_props["rho"])
            cp = self.xp.full((N, 1), default_props["cp"])

        x_coord = x[:, 0]
        y_coord = x[:, 1]

        for props in self.regions.values():
            # Now correctly comparing normalized inputs against normalized boundaries
            mask = (x_coord >= props["x_init_norm"]) & (x_coord <= props["x_fin_norm"]) & \
                   (y_coord >= props["y_init_norm"]) & (y_coord <= props["y_fin_norm"])
            mask = mask[..., None] 

            if self.backend == 'torch':
                k = self.xp.where(mask, self.xp.tensor(props["k"], device=x.device, dtype=k.dtype), k)
                rho = self.xp.where(mask, self.xp.tensor(props["rho"], device=x.device, dtype=rho.dtype), rho)
                cp = self.xp.where(mask, self.xp.tensor(props["cp"], device=x.device, dtype=cp.dtype), cp)
            else:
                k = self.xp.where(mask, props["k"], k)
                rho = self.xp.where(mask, props["rho"], rho)
                cp = self.xp.where(mask, props["cp"], cp)

        return {"k": k, "rho": rho, "cp": cp}
    
    def plot_geometry(self, type='smooth'):
        nx, ny = 50, 50

        x_1d = self.xp.linspace(self.X_MIN, self.X_MAX, nx)
        y_1d = self.xp.linspace(self.Y_MIN, self.Y_MAX, ny)
        X_norm, Y_norm = self.xp.meshgrid(x_1d, y_1d, indexing="ij")
        
        if self.backend == 'torch':
            coords = self.xp.stack([X_norm.ravel(), Y_norm.ravel(), self.xp.zeros_like(X_norm.ravel())], dim=1)
        else:
            coords = self.xp.stack([X_norm.ravel(), Y_norm.ravel(), self.xp.zeros_like(X_norm.ravel())], axis=1)
        if type == 'smooth':
            # Extract x and y columns from coords to match optimized function signature [N, 1]
            x_pts = coords[:, 0:1]
            y_pts = coords[:, 1:2]
            
            # Call the optimized function (returns a tuple)
            k_raw, rho_cp_raw = self.material_mask_regions_optimized(x_pts, y_pts)
            
            # Pack back into a dictionary to keep the rest of the function working
            props = {"k": k_raw, "rho_cp": rho_cp_raw}
        else:
            props = self.material_mask_regions_step(coords)

        if self.backend == 'torch':
            k_vals = props["k"].cpu().detach().numpy().reshape(nx, ny)
            X_plot = X_norm.cpu().detach().numpy()
            Y_plot = Y_norm.cpu().detach().numpy()
        else:
            k_vals = props["k"].reshape(nx, ny)
            X_plot = X_norm
            Y_plot = Y_norm

        # 3. Rescale purely for visualization
        # Formula: x_phys = (x_norm + 1) * lx / 2
        X_phys = (X_plot - self.X_MIN) * self.lx / (self.X_MAX - self.X_MIN)
        Y_phys = (Y_plot - self.Y_MIN) * self.ly / (self.Y_MAX - self.Y_MIN)

        plt.figure(figsize=(8, 3))
        plt.pcolormesh(X_phys, Y_phys, k_vals, shading='auto', cmap='viridis')
        plt.colorbar(label="Thermal conductivity k [W/m.K]")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"Material map (k) - {self.name}")
        plt.show()

    def hot_spot_volume(self):
        props = self.regions["active"]
        w_phys = abs(props["x_fin"] - props["x_init"])
        h_phys = abs(props["y_fin"] - props["y_init"])
        return w_phys * h_phys * 0.5e-3
