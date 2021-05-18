# we can create a robot
import numpy as np

import pendulum

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


us = [-5,0,5]

def cost(x,u):
    theta = x[0]
    theta_dot = x[1]
    res = (theta-np.pi)**2+0.01*theta_dot**2+0.0001*u**2
    return res



class Value_iter_solver:
    def __init__(self,robot, costfn = cost, actions=None, max_iters=100, sparse_loss=False):
        self.robot:pendulum = robot
        if actions is None:
            actions = [-5, 0, 5]
        self.action_list = np.array(actions)
        self.alpha = 0.9
        self.stop_loss = 1e-2
        self.max_iters = max_iters
        self.sparse_loss = sparse_loss
        self.costfn = costfn
        self.discretized_theta = discretized_theta
        self.discretized_theta_dot = discretized_thetadot
        self.space_shape = [len(discretized_theta),len(discretized_thetadot)]
        self.value = np.zeros(self.space_shape)
        self.policy = np.zeros(self.space_shape)
        self.make_state_transfer_table()

    def make_state_transfer_table(self):
        table_shape = [self.space_shape[0],self.space_shape[1],len(self.action_list),len(self.space_shape)]
        table = np.zeros(table_shape,dtype=np.int32)
        for i,theta in enumerate(self.discretized_theta):
            for j,v in enumerate(self.discretized_theta_dot):
                for k,u in enumerate(self.action_list):
                    next_state = self.robot.next_state([theta,v],u)
                    next_id = self.get_ids_by_state_value(next_state)
                    table[i,j,k,:] = np.array(next_id)
        self.state_transfer_table = table



    def get_state_by_ids(self,ids):
        thetaid,vid = ids[0],ids[1]
        theta = self.discretized_theta[thetaid]
        v = self.discretized_theta_dot[vid]
        return np.array([theta,v])

    def get_ids_by_state_value(self,x):
        theta,v = x[0],x[1]
        theta_id = np.argmin(np.abs(self.discretized_theta - theta))
        v_id = np.argmin(np.abs(self.discretized_theta_dot - v))
        return [theta_id,v_id]

    def value_iteration(self):
        J_last = np.zeros(self.space_shape)
        for i in range(self.max_iters):
            J_new = np.zeros_like(J_last)
            for j in range(self.space_shape[0]):
                for k in range(self.space_shape[1]):
                    score = np.zeros(len(self.action_list))
                    for l, u in enumerate(self.action_list):

                        theta = self.discretized_theta[j]
                        dot = self.discretized_theta_dot[k]

                        next_state_id = self.state_transfer_table[j,k,l,:]

                        score_next = J_last[next_state_id[0],next_state_id[1]]
                        cost_score = self.costfn([theta,dot],u)

                        score[l] = cost_score + self.alpha * score_next
                    J_new[j,k] = min(score)
                    self.policy[j,k] = self.action_list[np.argmin(score)]
            delta =  (J_new-J_last)
            max_delta = np.abs(delta).max()
            print(max_delta)
            if max_delta<self.stop_loss:
                print("CONVERGED after iteration " + str(i))
                self.value = J_new
                break
            else:
                J_last = J_new
        return self.value,self.policy






solver = Value_iter_solver(robot,costfn=cost)
res = solver.value_iteration()