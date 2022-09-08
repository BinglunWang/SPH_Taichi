import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = -9.80  # Gravity
        self.viscosity = 0.01  # viscosity
        self.density_0 = 1000.0  # reference density
        self.dt = ti.field(float, shape=())
        self.dt[None] = 2e-4

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    def initialize_solver(self):
        self.ps.initialize_particle_system()
        self.compute_rigid_rest_cm()
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()

    @ti.kernel
    def compute_rigid_rest_cm(self):
        self.ps.rigid_rest_cm[None] = self.compute_cos()

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_boundary:
                continue
            x_i = self.ps.x[p_i]
            delta = self.cubic_kernel(0.0)
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                delta += self.cubic_kernel((x_i - x_j).norm())
            self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly
            # print(self.ps.m_V0, " ", self.ps.m_V[p_i])
    

    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_moving_rigid_body:
                continue
            x_i = self.ps.x[p_i]
            delta = self.cubic_kernel(0.0)
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                delta += self.cubic_kernel((x_i - x_j).norm())
            self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly
            # print(self.ps.m_V0, " ", self.ps.m_V[p_i])

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[p_i] -= (
            1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type:int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == particle_type: # self.ps.material_fluid or self.ps.material[p_i] == self.ps.material_moving_rigid_body:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > self.ps.bound[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.bound[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.bound[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.bound[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_3D(self, particle_type:int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == particle_type: #self.ps.material_fluid or self.ps.material[p_i] == self.ps.material_moving_rigid_body:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.bound[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.bound[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.bound[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.bound[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding

                if pos[2] > self.ps.bound[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x[p_i][2] = self.ps.bound[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x[p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)


    # TODO: Reduce optimization
    @ti.func
    def compute_cos(self):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_moving_rigid_body:
                # mass = self.ps.m_V[p_i]
                mass = self.ps.m_V0 * self.ps.density[p_i]
                cm += mass * self.ps.x[p_i]
                sum_m += mass
        cm /= sum_m
        # print(cm)
        return cm

    @ti.kernel
    def solve_constraints(self):
        # compute center of mass
        cm = self.compute_cos()
        # A
        A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_moving_rigid_body:
                q = self.ps.x_0[p_i] - self.ps.rigid_rest_cm[None]
                p = self.ps.x[p_i] - cm
                # A += self.ps.m_V[p_i] * p @ q.transpose()
                A += self.ps.m_V0 * self.ps.density[p_i] * p @ q.transpose()

        R, S = ti.polar_decompose(A)
        
        if all(abs(R) < 1e-6):
            R = ti.Matrix.identity(ti.f32, 3)
        
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_moving_rigid_body:
                goal = cm + R @ (self.ps.x_0[p_i] - self.ps.rigid_rest_cm[None])
                corr = (goal - self.ps.x[p_i]) * 1.0
                self.ps.x[p_i] += corr
        

    @ti.kernel
    def compute_rigid_collision(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_moving_rigid_body:
                continue
            cnt = 0
            x_delta = ti.Vector([0.0 for i in range(self.ps.dim)])
            # v_delta = ti.Vector([0.0 for i in range(self.ps.dim)])
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]

                if self.ps.material[p_j] == self.ps.material_boundary:
                    cnt += 1
                    x_j = self.ps.x[p_j]
                    r = self.ps.x[p_i] - x_j
                    if r.norm() < self.ps.particle_diameter:
                        x_delta += (r.norm() - self.ps.particle_diameter) * r.normalized()
                        # v_delta -= 1.5 * self.ps.v[p_i].dot(r.normalized()) * r.normalized()
                        # self.ps.acceleration[p_i] += 10000 * (r.norm() - self.ps.particle_diameter) * r.normalized()
            if cnt > 0:
                self.ps.x[p_i] += 2.0 * x_delta / cnt
                # self.ps.v[p_i] += v_delta / cnt
                        


    def solve_rigid_body(self):
        for i in range(5):
            self.solve_constraints()
            self.compute_rigid_collision()
            self.enforce_boundary_3D(self.ps.material_moving_rigid_body)


    def step(self):
        self.ps.initialize_particle_system()
        # self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()
        if self.ps.dim == 2:
            self.enforce_boundary_2D(self.ps.material_fluid)
        elif self.ps.dim == 3:
            self.enforce_boundary_3D(self.ps.material_fluid)
