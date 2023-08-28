import math
import matplotlib.pyplot as plt
import numpy as np
import control

import time

import pathlib
import sys


sys.path.append(str(pathlib.Path(__file__).parent.parent))
from CBFsteer import CBF_RRT
import env, plotting, utils


from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are

SHOW_ANIMATION = False


class LQRPlanner:

    def __init__(self):
        '''
        Create planner for quadcotpers

        TO DO: (1) 3d LQR Planning on linearized drone dynamics
               (2) CBF with linearized drone dynamcis
               (3) intergrate with MPC (Could be linear or nonlinear)
        '''

        self.N = 12  # number of state variables
        self.M = 4  # number of control variables
        self.DT = 0.1  # discretization step

        self.MAX_TIME = 2.5  # Maximum simulation time
        self.GOAL_DIST = 0.5
        self.MAX_ITER = 10
        self.EPS = 0.01

        # initialize CBF
        self.env = env.Env()
        #self.obs_circle = self.env.obs_circle
        #self.obs_rectangle = self.env.obs_rectangle
        self.obs_sphere = []
        self.obs_box = [] # fourth order poly-nomial approximation
        self.obs_boundary = self.env.obs_boundary

        #self.cbf_rrt_simulation = CBF_RRT(self.obs_circle)

    def drone_nonlinear(self, x, t, args=None):
        # Drone parameters

        m = 0.028  # mass in kg
        g = 9.81  # gravity
        I_xx = 16.571710e-6  # Moment of inertia around x-axis
        I_yy = 16.655602e-6  # Moment of inertia around y-axis
        I_zz = 29.261652e-6  # Moment of inertia around z-axis
        '''

        m = 0.068  # mass in kg
        g = 9.81  # gravity
        I_xx = 6.89e-5  # Moment of inertia around x-axis
        I_yy = 6.89e-5  # Moment of inertia around y-axis
        I_zz = 1.366e-4  # Moment of inertia around z-axis
        '''
        u = args
        x1, x2, y1, y2, z1, z2, phi1, phi2, theta1, theta2, psi1, psi2 = x.reshape(-1).tolist()
        #ft, tau_x, tau_y, tau_z = u.reshape(-1).tolist() # ft:trust in z, tau: moment in x,y,z
        # uppack control
        ft = u[0][0]
        tau_x = u[1][0]
        tau_y = u[2][0]
        tau_z = u[3][0]

        dot_x1 = x2
        dot_x2 = ft / m * (np.sin(phi1) * np.sin(psi1) + np.cos(phi1) * np.cos(psi1) * np.sin(theta1))
        dot_y1 = y2
        dot_y2 = ft / m * (np.cos(phi1) * np.sin(psi1) * np.sin(theta1) - np.cos(psi1) * np.sin(phi1))
        dot_z1 = z2
        dot_z2 = -g + ft / m * np.cos(phi1) * np.cos(theta1)
        dot_phi1 = phi2
        dot_phi2 = (I_yy - I_zz) / I_xx * theta2 * psi2 + tau_x / I_xx
        dot_theta1 = theta2
        dot_theta2 = (I_zz - I_xx) / I_yy * phi2 * psi2 + tau_y / I_yy
        dot_psi1 = psi2
        dot_psi2 = (I_xx - I_yy) / I_zz * phi2 * theta2 + tau_z / I_zz

        return [dot_x1, dot_x2, dot_y1, dot_y2, dot_z1, dot_z2, dot_phi1, dot_phi2, dot_theta1, dot_theta2, dot_psi1, dot_psi2]

    def lqr_planning(self, current_state, goal_state, LQR_gain=None, test_LQR=False, show_animation=True, cbf_check=False,
                     solve_QP=False):

        # Drones with 12 dimensions, current_state is the state of the drone at tk,
        # goal_state is the local goal needs to be reached


        # crazyflie
        m = 0.028  # mass in kg
        g = 9.81  # gravity
        I_xx = 16.571710e-6  # Moment of inertia around x-axis
        I_yy = 16.655602e-6  # Moment of inertia around y-axis
        I_zz = 29.261652e-6  # Moment of inertia around z-axis

        xk = np.array(current_state).reshape(self.N, 1)
        xd = np.array(goal_state).reshape(self.N, 1)
        ud = np.matrix([[m*g], [0], [0], [0]])

        # Crazyflie reference
        # https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/params/#cppmrateroll

        # Caterteian position constraint
        x_max, y_max, z_max = 1.5, 1.5, 1.5
        vx_max, vy_max, vz_max = 1.0, 1.0, 1.0

        # Attitude constraint
        phi_max, theta_max, psi_max = np.pi/4, np.pi/4, np.pi/2
        phidot_max, thetadot_max, psidot_max = np.pi*4, np.pi*4, np.pi*2.22

        # Control constraint
        f_max = 0.6 # N
        moment_z_max =  (0.005964552 * (0.6/4) + 1.563383e-5)*4
        moment_x_max, moment_y_max = (0.6/4)*0.046, (0.6/4)*0.046

        # Q weight matrix obtained from Bryson's rule
        Q = np.array([
            [1/(0.05*x_max**2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1/(0.05*vx_max**2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # vx
            [0, 0, 1/(0.05*y_max**2), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 0, 1/(0.05*vy_max**2), 0, 0, 0, 0, 0, 0, 0, 0],  # vy
            [0, 0, 0, 0, 10/(0.05*z_max**2), 0, 0, 0, 0, 0, 0, 0],  # z
            [0, 0, 0, 0, 0, 1/(0.05*vz_max**2), 0, 0, 0, 0, 0, 0],  # vz
            [0, 0, 0, 0, 0, 0, 1/(0.05*phi_max**2), 0, 0, 0, 0, 0],  # phi
            [0, 0, 0, 0, 0, 0, 0, 1/(0.05*phidot_max**2), 0, 0, 0, 0],  # phidot
            [0, 0, 0, 0, 0, 0, 0, 0, 1/(0.05*theta_max**2), 0, 0, 0],  # theta
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1/(0.05*thetadot_max**2), 0, 0],  # thetadot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/(0.05*psi_max**2), 0],  # psi
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/(0.05*psidot_max**2)]  # psidot
        ])

        R = np.diag([1/(f_max**2), 1/(moment_x_max**2), 1/(moment_y_max**2), 1/(moment_z_max**2)])

        # A matrix (12x12)
        A = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        # B matrix (12x4)
        B = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1 / m, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1 / I_xx, 0, 0],
            [0, 0, 0, 0],
            [0, 0 , 1 / I_yy, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1 / I_zz]
        ])


        self.K, _, _ = control.lqr(A, B, Q, R)
        #discrete_system = control.ss(A, B, np.eye(12), np.zeros((12, 4)), self.DT)
        #self.K, _, _ = control.dlqr(discrete_system.A, discrete_system.B, Q, R)
        #self.K, _, _ = control.lqr(A, B, Q, R)

        rx, ry, rz = [xk[0,0]], [xk[2,0]], [xk[4,0]]

        error = []

        found_path = False

        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            x_error = xk - xd
            u = -self.K @ x_error + np.array([[m*g],[0],[0],[0]])
            #u = np.array([[m*g+1.0],[0],[0],[0]])

            # check if LQR control is safe with respect to CBF constraint
            '''
            if cbf_check and not test_LQR and not solve_QP:
                if not self.cbf_rrt_simulation.QP_constraint([x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gtheta], u,
                                                             system_type="unicycle_velocity_control"):
                    break
            '''
            #u_clipped = np.clip(u, -np.array([[f_max], [moment_x_max], [moment_y_max], [moment_z_max]]),
            #                    np.array([[f_max], [moment_x_max], [moment_y_max], [moment_z_max]]))
            # Intergrate the nonlinear dynamics

            #print(odeint(self.drone_nonlinear, xk_rollout, [0, self.DT], args=(u,)))
            xk = odeint(self.drone_nonlinear, xk.reshape(-1), [0, self.DT], args=(u,))[-1].reshape(-1, 1)


            rx.append(xk[0, 0])
            ry.append(xk[2, 0])
            rz.append(xk[4, 0])

            d = math.sqrt((goal_state[0] - rx[-1]) ** 2 + (goal_state[2] - ry[-1]) ** 2 + (goal_state[4] - rz[-1]) ** 2)
            error.append(d)

            if d <= self.GOAL_DIST:
                found_path = True
                # print('errors ', d)
                break

            # animation
            if show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.

                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(0) if event.key == 'escape' else None])
                #plt.plot(sx, sy, sz, "or")
                #plt.plot(gx, gy, gz,"ob")
                plt.plot(rx, ry, rz,"-r")
                plt.axis("equal")
                plt.pause(1.0)

        if not found_path:
            # print("Cannot found path")
            return rx, ry, rz, error, found_path

        return rx, ry, rz, error, found_path

    def dLQR(self, A, B, Q, R):

        N = 50

        # Create a list of N + 1 elements
        P = [None] * (N + 1)

        Qf = Q

        # LQR via Dynamic Programming
        P[N] = Qf

        # For i = N, ..., 1
        for i in range(N, 0, -1):
            # state cost matrix
            P[i - 1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

            # Create a list of N elements
        K = [None] * N
        u = [None] * N

        P1 = P[N - 1]

        K1 = -np.linalg.inv(R + B.T @ P[N] @ B) @ B.T @ P[N] @ A

        return K1

    def get_linear_model(self, x_bar, u_bar):
        """
        Computes the LTI approximated state space model x' = Ax + Bu

        State Space dimension 12 x 1

        x^T = [x, x_dot, y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot]

        We change the notation to be (equivalent to the above)
        x^T = [x1, x2, y1, y2, z1, z2, phi1, phi2, theta1, theta2, psi1, psi2]

        F      - 1 x 1, thrust output in z direction
        M      - 3 x 1, [\tau_x, \tau_y, \tau_z] moments output

        Control Space dimension 4 x 1
        u^T = [F,M]

        """

        x1 = x_bar[0]
        x2 = x_bar[1]
        y1 = x_bar[2]
        y2 = x_bar[3]
        z1 = x_bar[4]
        z2 = x_bar[5]
        phi1 = x_bar[6]
        phi2 = x_bar[7]
        theta1 = x_bar[8]
        theta2 = x_bar[9]
        psi1 = x_bar[10]
        psi2 = x_bar[11]

        F = u_bar[0]
        M_x = u_bar[1]
        M_y = u_bar[2]
        M_z = u_bar[3]

        # Drone parameters
        m = 0.028  # mass in kg
        g = 9.81  # gravity
        I_xx = 16.571710e-6  # Moment of inertia around x-axis
        I_yy = 16.655602e-6  # Moment of inertia around y-axis
        I_zz = 29.261652e-6  # Moment of inertia around z-axis

        # A matrix (12x12)
        A = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        # B matrix (12x4)
        B = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/m, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1/I_xx, 0, 0],
            [0, 0, 0, 0],
            [0, 1/I_yy, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1/I_zz, 0]
        ])


        A_lin = np.eye(self.N) + self.DT * A

        B_lin = self.DT * B

        f_xu = np.array(
            [
                x2,
                F / m * (np.sin(phi1) * np.sin(psi1) + np.cos(phi1) * np.cos(psi1) * np.sin(theta1)),
                y2,
                F / m * (np.cos(phi1) * np.sin(psi1) * np.sin(theta1) - np.cos(psi1) * np.sin(phi1)),
                z2,
                -g + F / m * np.cos(phi1) * np.cos(theta1),
                phi2,
                (I_yy - I_zz) / I_xx * theta2 * psi2 + M_x / I_xx,
                theta2,
                (I_zz - I_xx) / I_yy * phi2 * psi2 + M_y / I_yy,
                psi2,
                (I_xx - I_yy) / I_zz * phi2 * theta2 + M_z / I_zz]
        ).reshape(self.N, 1)

        C_lin = self.DT * (
                f_xu - np.dot(A, x_bar.reshape(self.N, 1)) - np.dot(B, u_bar.reshape(self.M, 1))
        )

        return np.round(A_lin, 4), np.round(B_lin, 4), np.round(C_lin, 4)



def main():
    print(__file__ + " start!!")

    max_steps = 10

    gx = 0.3
    gy = 0.2
    gz = 1.5


    lqr_planner = LQRPlanner()

    sx, sy, sz = 0.0, 0.0, 0.0
    initial_state = [sx,0.0,sy,0.0,sz,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    goal_state = [gx, 0.0, gy, 0.0, gz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    start_time = time.time()

    print("goal", gy, gx, gz)


    rx, ry, rz, error, foundpath = lqr_planner.lqr_planning(initial_state, goal_state, LQR_gain=None, test_LQR=True,
                                                                show_animation=SHOW_ANIMATION)


    print("time of running LQR: ", time.time() - start_time)
    print("Found path: ", foundpath)

    ax = plt.axes(projection='3d')
    ax.plot(sx, sy, sz, "or")
    ax.plot(gx, gy, gz, "ob")
    ax.plot(rx, ry, rz, "-r")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-5, 5)
    plt.show()

    '''
        
        ax1.plot(sx, sy, "or")
        ax1.plot(gx, gy, "ob")
        ax1.plot(rx, ry, "-r")
        ax1.grid()

        ax2.plot(error, label="errors")
        ax2.legend(loc='upper right')
        ax2.grid()
        plt.show()

        if SHOW_ANIMATION:  # pragma: no cover
            plt.plot(sx, sy, "or")
            plt.plot(gx, gy, "ob")
            plt.plot(rx, ry, "-r")
            plt.axis("equal")
            plt.pause(1.0)
    '''

if __name__ == '__main__':
    main()
