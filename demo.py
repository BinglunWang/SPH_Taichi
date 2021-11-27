import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from sph_base import WCSPHSolver
ti.init(arch=ti.cpu, device_memory_GB=1)

if __name__ == "__main__":
    ps = ParticleSystem((512, 512))

    # ps.add_cube(lower_corner=[3, 1],
    #             cube_size=[1.0, 1.0],
    #             velocity=[0.0, 12.0],
    #             density=1000.0,
    #             color=0xFFFFF0,
    #             material=1)

    ps.add_cube(lower_corner=[6, 2],
                cube_size=[1.0, 1.0],
                velocity=[0.0, 10.0],
                density=1000.0,
                color=0xFFFFF0,
                material=1)

    ps.add_cube(lower_corner=[3, 4],
                cube_size=[5.0, 5.0],
                velocity=[0.0, -10.0],
                density=1000.0,
                color=0xEEEEE0,
                material=1)

    wcsph_solver = WCSPHSolver(ps)
    gui = ti.GUI()
    while gui.running:
        for i in range(5):
            wcsph_solver.step()

        particle_info = ps.dump()
        # print(particle_info['position'])

        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 2 * ps.screen_to_world_ratio,
                    color=0x956333)
        # print(particle_info['position'][660, :], particle_info['position'][1550, :], particle_info['position'][2000, :])
        # gui.circles(np.array([particle_info['position'][0, :]]) * ps.screen_to_world_ratio / 512,
        #             radius=ps.particle_radius / 2 * ps.screen_to_world_ratio,
        #             color=0xFFFFFF)
        gui.circles(np.array([particle_info['position'][25, :]]) * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 2 * ps.screen_to_world_ratio,
                    color=0xFFFFFF)
        gui.circles(np.array([particle_info['position'][75, :]]) * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 2 * ps.screen_to_world_ratio,
                    color=0xFFFFFF)
        gui.show()