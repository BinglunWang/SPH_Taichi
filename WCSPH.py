import taichi as ti
from sph_base import SPHBase

class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # P = k1*((rho_i/rho^0)^k2 - 1), k2 is exponent
        self.exponent = 7.0
        self.exponent = self.ps.cfg.get_cfg('exponent')

        # stiffness
        self.stiffness = 50000.0
        self.stiffness = self.ps.cfg.get_cfg('stiffness')
        
        # surface tension
        self.surface_tension = 0.01
        self.dt[None] = self.ps.cfg.get_cfg('timeStepSize')


    # rho 
    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        # Explicit Volume Deviation rho^0 * sum of m Wij but with normlised
        if self.ps.material[p_j] == self.ps.material_fluid:
            # neighors
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary solid [Akinci et al., TOG12] ?? why same as above
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())

    @ti.kernel
    def compute_densities(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            self.ps.density[p_i] *= self.density_0
    
    
    @ti.func
    def compute_pressure_forces_task(self, p_i, p_j, ret:ti.template()):
        # aim to get dv derivate of pressure etc. (already calculated pressure)
        x_i = self.ps.x[p_i]
        dpi = self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
        
        # -derivative of pressure)u / rho_i:  -rho^0 * m * (Pi/ rhoi^2 + Pj/rhoj^2) derivative of Wij 
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            density_j = self.ps.density[p_j] * self.density_0 / self.density_0 # (It's a To do)
            dpj = self.ps.pressure[p_j] / (density_j * density_j)
            ret += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) * self.cubic_kernel_derivative(x_i-x_j)

        elif self.ps.material[p_j] == self.ps.material_solid:
            # [Akinci et al., TOG12] ?? why same as above
            dpj = self.ps.pressure[p_i] / self.density_0 ** 2
            x_j = self.ps.x[p_j]
            f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) * self.cubic_kernel_derivative(x_i-x_j)
            ret += f_p
            # different:
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -f_p  * self.density_0 / self.ps.density[p_j]

    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # with clamped rhoi = maxA(rhoi - rho^0, 0)
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
            # k1*((rho_i/rho^0)^k2 - 1)
            self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i):
                continue
            dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dv)
            self.ps.acceleration[p_i] += dv


    ## ?? surface tension and viscosity, need to know the details
    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]

        ## surface tension ##
        if self.ps.material[p_j] == self.ps.material_fluid:
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            # if two particles are close:  - tension /  mi * mj * r * wij/|wij| or wd/|wd|
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
            
        ## viscosity force ##
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        # neighor is fluid: d * viscosity * (mj / rho_j) * (vi - vj) * r / ((r/|r|)^2 + 0.01 * r^2) * derivative of (Wr) 
        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(r)
            ret += f_v

        # solid: update a by f_v * rho^0 / rho_j
        elif self.ps.material[p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)

            # f_v = 0
            ret += f_v

            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]


    # compute non pressure forces for fluid particles with viscosity and surface tension. Fluid and Solid both have gravity
    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            # gravity
            d_v = ti.Vector(self.g) 
            self.ps.acceleration[p_i] = d_v
            # viscosity and surface tension
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[p_i] = d_v



    @ti.kernel
    def advect(self):
        # semi-implict Euler
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    
    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()