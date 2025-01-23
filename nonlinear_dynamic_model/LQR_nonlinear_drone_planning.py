import numpy as np
from scipy.integrate import odeint
import control
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DroneParams:
    mass: float = 0.028
    gravity: float = 9.81
    I_xx: float = 16.571710e-6
    I_yy: float = 16.655602e-6
    I_zz: float = 29.261652e-6


class LQRController:
    def __init__(self):
        self.drone = DroneParams()
        self.N = 12  # state dimension
        self.M = 4  # control dimension

        self.A = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # phi
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # theta
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # psi
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # p
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # q
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # r
                [0, self.drone.gravity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x_dot
                [-self.drone.gravity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y_dot
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # z_dot
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # x
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # y
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # z
            ]
        )

        self.B = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1 / self.drone.I_xx, 0, 0],
                [0, 0, 1 / self.drone.I_yy, 0],
                [0, 0, 0, 1 / self.drone.I_zz],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1 / self.drone.mass, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        # LQR gains
        self.Q = np.diag([100, 100, 100, 10, 10, 10, 10, 10, 10, 1, 1, 1])
        self.R = np.diag([0.1, 0.1, 0.1, 0.1])
        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)

    def get_control(self, current_state, goal_state):
        x_error = current_state.reshape(self.N, 1) - goal_state.reshape(self.N, 1)
        ud = np.array([[self.drone.mass * self.drone.gravity, 0, 0, 0]]).T
        u = -self.K @ x_error + ud
        return u.flatten()


class DroneSimulation:
    def __init__(self):
        self.drone = DroneParams()
        self.dt = 0.02
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.states_history = []

    def dynamics(self, x, t, u):
        phi, theta, psi, p, q, r, vx, vy, vz, x_pos, y_pos, z_pos = x
        ft, tau_x, tau_y, tau_z = u

        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        t_theta = np.tan(theta)

        dx = np.zeros(12)
        # Angular velocities
        dx[0] = p + q * s_phi * t_theta + r * c_phi * t_theta
        dx[1] = q * c_phi - r * s_phi
        dx[2] = q * s_phi / c_theta + r * c_phi / c_theta

        # Angular accelerations
        dx[3] = (
            (self.drone.I_yy - self.drone.I_zz) * q * r
        ) / self.drone.I_xx + tau_x / self.drone.I_xx
        dx[4] = (
            (self.drone.I_zz - self.drone.I_xx) * p * r
        ) / self.drone.I_yy + tau_y / self.drone.I_yy
        dx[5] = (
            (self.drone.I_xx - self.drone.I_yy) * p * q
        ) / self.drone.I_zz + tau_z / self.drone.I_zz

        # Linear velocities
        dx[6] = (c_phi * s_theta * c_psi + s_phi * s_psi) * ft / self.drone.mass
        dx[7] = (c_phi * s_theta * s_psi - s_phi * c_psi) * ft / self.drone.mass
        dx[8] = -self.drone.gravity + (c_phi * c_theta) * ft / self.drone.mass

        # Positions
        dx[9] = vx
        dx[10] = vy
        dx[11] = vz

        return dx

    def step(self, state, action):
        next_state = odeint(self.dynamics, state, [0, self.dt], args=(action,))[-1]
        self.states_history.append(next_state.copy())
        return next_state

    def render(self, state):
        self.ax.clear()

        phi, theta, psi = state[0:3]
        x, y, z = state[9:12]

        # Plot drone position
        self.ax.scatter(x, y, z, c="r", marker="o", s=1)

        # X configuration arms
        arm_length = 0.3
        arms = np.array(
            [
                [arm_length / np.sqrt(2), arm_length / np.sqrt(2), 0],
                [-arm_length / np.sqrt(2), -arm_length / np.sqrt(2), 0],
                [arm_length / np.sqrt(2), -arm_length / np.sqrt(2), 0],
                [-arm_length / np.sqrt(2), arm_length / np.sqrt(2), 0],
            ]
        )

        # Rotation matrix
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        )

        Ry = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

        Rz = np.array(
            [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
        )

        R = Rz @ Ry @ Rx
        arms_rotated = np.dot(arms, R.T) + np.array([x, y, z])

        # Draw X configuration
        for i in range(0, 4, 2):
            self.ax.plot(
                [arms_rotated[i][0], arms_rotated[i + 1][0]],
                [arms_rotated[i][1], arms_rotated[i + 1][1]],
                [arms_rotated[i][2], arms_rotated[i + 1][2]],
                c="b",
            )
            self.ax.plot(
                [arms_rotated[i + 1][0], arms_rotated[(i + 2) % 4][0]],
                [arms_rotated[i + 1][1], arms_rotated[(i + 2) % 4][1]],
                [arms_rotated[i + 1][2], arms_rotated[(i + 2) % 4][2]],
                c="b",
            )

        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([0, 5])

        plt.draw()
        plt.pause(0.02)


def main():
    controller = LQRController()
    sim = DroneSimulation()

    current_state = np.zeros(12)
    goal_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 2.5])

    for _ in range(500):
        # Get control action
        u = controller.get_control(current_state, goal_state)

        # Simulate one step
        current_state = sim.step(current_state, u)

        # Render
        sim.render(current_state)

        # Check convergence
        if np.linalg.norm(current_state[9:12] - goal_state[9:12]) < 0.5:
            print("Reached goal!")
            break

    plt.show()


if __name__ == "__main__":
    main()
