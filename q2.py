# we can create a robot
import numpy as np

import pendulum
import tqdm
robot = pendulum.Pendulum()

# assume we set theta and dtheta = 0 and u = -5, we can get the next state using
x = np.array([0,0])
u = -5
x_next = robot.next_state(x, u)

# we don't want 2pi to be in the set because it's the same as 0
discretized_theta = np.linspace(0, 2*np.pi, 50, endpoint=False)

discretized_thetadot = np.linspace(-6, 6, 50)

# now given an arbitrary continuous state theta
theta_arbitrary = 0.23471

# we can find the index of the closest element in the set of discretized states
index_in_discretized_theta = np.argmin(np.abs(discretized_theta - theta_arbitrary))

# and find the closed discretized state
closest_state = discretized_theta[index_in_discretized_theta]
print(f'the discretized theta closest to {theta_arbitrary} is {closest_state} with index {index_in_discretized_theta}')

def cost(x,u):
    theta = x[0]
    theta_dot = x[1]
    res = (theta-np.pi)**2+0.01*theta_dot**2+0.0001*u**2
    return res

class Q_solver:
    def __init__(self,robot, costfn = cost, actions=None, max_iters=400, sparse_loss=False):
        self.robot: pendulum = robot
        if actions is None:
            actions = [-5, 0, 5]
        self.action_list = np.array(actions)
        self.lr = 0.1
        self.epslion = 0.1
        self.step_in_iter = 2000
        self.alpha = 0.95
        self.max_iters = max_iters
        self.sparse_loss = sparse_loss
        self.costfn = costfn
        self.discretized_theta = discretized_theta
        self.discretized_theta_dot = discretized_thetadot
        self.space_shape = [len(discretized_theta),len(discretized_thetadot)]

        self.num_states = 50*50
        self.nu = 3
        self.nq = 50
        self.make_state_transfer_table()
    def make_state_transfer_table(self):
        next_state_index = np.empty([self.num_states, self.nu], dtype=np.int32)
        for i in range(self.num_states):
            for k in range(self.nu):
                x_next = robot.next_state(self.get_states(i), self.action_list[k])
                next_state_index[i, k] = self.get_index(x_next)

        self.state_transfer_table = next_state_index #[250 3 2]


    def get_index(self,x):
        ind_q = np.argmin((x[0] - self.discretized_theta) ** 2)
        ind_v = np.argmin((x[1] - self.discretized_theta_dot) ** 2)
        return ind_q + ind_v * self.nq

    def get_states(self, index):
        iv, ix = np.divmod(index, self.nq)
        return np.array([self.discretized_theta[ix], self.discretized_theta_dot[iv]])


    def iterate(self):
        q = np.zeros([self.num_states, self.nu])
        q_Last = np.zeros([self.num_states, self.nu])
        for i in tqdm.tqdm(range(self.max_iters)):  #

            # choose initial state x0
            x_0 = np.array([0, 0])
            x_index = self.get_index(x_0)
            for j in range(self.step_in_iter):

                if np.random.uniform(0, 1) > self.epslion:
                    u_index = np.argmin(q[x_index, :])
                else:
                    u_index = np.random.randint(0, self.nu - 1)
                # observe x_t+1
                next_index = self.state_transfer_table[x_index, u_index]
                # compute g(x_t,u(x_t))
                x = self.get_states(x_index)
                u = self.action_list[u_index]
                # compute TDerror
                TDerror = self.costfn(x, u) + self.alpha * min(q[next_index, :]) - q[
                    x_index, u_index]
                q[x_index, u_index] = q[x_index, u_index] + self.lr * TDerror
                x_index = next_index

            # we update the current Q function if there is any change otherwise we are done
            if ((q_Last - q) ** 2 < 1e-2).all():
                break
            else:
                q_Last = q.copy()

        policy = np.zeros(self.space_shape)
        value_function = np.zeros(self.space_shape)
        for k in range(self.num_states):
            iv, ix = np.divmod(k, self.nq)
            policy[ix,iv] = self.action_list[np.argmin(q[k, :])]
            value_function[ix,iv] = min(q[k, :])
        return value_function,policy
solver = Q_solver(robot,cost)
value,policy = solver.iterate()


