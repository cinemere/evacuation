import numpy as np
import gym
from gym import spaces

import os

from collections import namedtuple
import numpy as np
from math import sqrt
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import imageio
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)

Point = namedtuple('Point', 'x, y')


class GameEnv(gym.Env):
    """
    Evacuation Game Enviroment for Gym
    Continious Action and Observation Space
    """

    def __init__(self, 
                NUM_PEDESTRIANS=10,             # number of pedestrians
                WIDTH=1.0, HEIGHT=1.0,          # geometry of environment space
                VISION_RADIUS=0.2,              # radius of catch by leader
                ALPHA = 3,                      # parameter of gradient state
                STEP_SIZE = 0.01):  # 0.1       # step_size
        super(GameEnv, self).__init__()

        # action and observation space
        NUM_STATES = 6 # 14
        NUM_ACTIONS = 2 
        self.action_space = spaces.Box(low=-1, high=1, shape=(NUM_ACTIONS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(NUM_STATES,), dtype=np.float32)

        # simulation area
        self.eps = np.finfo(np.float32).eps.item()
        self.alpha = ALPHA
        self.w = WIDTH
        self.h = HEIGHT
        self.vision_radius = VISION_RADIUS
        self.exit_radius = 0.4
        self.evacuation_radius = 0.01
        self.step_size = STEP_SIZE

        # agent
        self.agent_position = None
        self.agent_direction = None

        # pedestrians
        self.N_pedestrians = NUM_PEDESTRIANS
        self.pedestrians_position = None
        self.pedestrians_direction = None

        # pedestrians' status
        self.status_viscek = None
        self.status_catched = None
        self.status_exiting = None
        self.status_evacuated = None

        # plotting variables
        self.walk_diagram = None
        self.image_array = None
        self.diagram_log = None

        # parameters for reset (start of episode)
        self.t = None
        self.tmax = 200 * self.N_pedestrians
        self.reward = None
        self.sum_distance = None
        self.sum_distance_at_start = None

        self.reset()

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # init game state
        self.t = 0
        self.reward = 0

        # positions of exit, leader ane pedestrians
        self._place()

        # status of each pedestrian
        self._status()

        # beginning distance                
        self.sum_distance = self._sum_distance()
        self.sum_distance_at_start = self.sum_distance

        # plotting variable
        self.walk_diagram = [self._return_positions()]
        self.image_array = []
        self.diagram_log = []
        self._log_diagram()

        self.baseline = self._create_baseline()

        return self._return_state()

    def step(self, action):
        self.reward = 0
        self.t += 1

        # Leader and pedestrians perform movement
        # and reward for movement
        self._move(action)
        self._log_diagram()

        # Calc score and reward for this step
        self.reward += 1 - self._sum_distance()

        # Check termination condition
        game_over = False

        if sum(self.status_evacuated) == self.N_pedestrians:
            # Termination due to all pedestrians saved
            print("S", end=" ")
            game_over = True
            self.reward = self.tmax - self.t

        if self.t > self.tmax:  # (self.h + self.w) * self.N_pedestrians:
            # Termination due to finished time
            print("F{}".format(int(sum(self.status_evacuated))), end=" ")
            game_over = True
            self.reward = self.tmax * (sum(self.status_evacuated) / self.N_pedestrians - 1)
            # self.reward = 300 * (sum(self.status_evacuated) / self.N_pedestrians) - self._sum_distance() - self.t

        state = self._return_state()

        return state, self.reward, game_over, {}

    def create_walk_diagram(self, mode='one_image', path='model/images/n_episode/', n_episode=0):

        positions = self._return_positions()
        self.walk_diagram.append(positions)
        if mode == 'animation':
            if self.t % 5 == 0:
                self.image_array.append(self.plot_walk_diagram(path, n_episode))

    def plot_walk_diagram(self, path='model/images/', n_episode=0):

        # read pedestrians positions
        states = np.array(self.walk_diagram)

        fig, ax = plt.subplots(figsize=(5, 5))

        # plot exit zone
        exiting_circle = mpatches.Wedge((self.exit[0], self.exit[1]), self.exit_radius, 0, 180, alpha = 0.2, color='green')
        saving_circle = mpatches.Wedge((self.exit[0], self.exit[1]), self.evacuation_radius, 0, 180, color='white')
        ax.add_patch(exiting_circle)
        ax.add_patch(saving_circle)
        ax.plot(self.exit[0], self.exit[1], marker='X', color='green')    
 
        # plot leader trajectory
        ax.plot(states[:, 0, 0], states[:, 0, 1], color='black', alpha=0.6, linewidth=1)
        ax.plot(states[-1, 0, 0], states[-1, 0, 1], color='black', marker='2', ms = 1)

        # plot pedestrians trajectories
        for pedestrian_id in range(self.N_pedestrians):

            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(states[:, pedestrian_id + 1, 0], states[:, pedestrian_id + 1, 1], alpha=0.1, linewidth=3, color=color)
            ax.plot(states[-1, pedestrian_id + 1, 0], states[-1, pedestrian_id + 1, 1], marker='.', color=color)

        plt.xlim([-1.1, 1.1]), plt.ylim([-1.1, 1.1]) 
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.text(1, 1, str(sum(self.status_evacuated)))
        plt.title('{}{:04d}'.format(path, n_episode))
        plt.tight_layout()

        # draw plot for pixel transformation
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        output = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return output

    def render(self, mode='console', path='model/images/n_episode/', n_episode=0):
        print('reward', self.reward)
        print('saved', sum(self.status_evacuated))

    def record(self, mode='one_image', path='model/images/n_episode/', n_episode=0):

        if mode == 'one_image':
            # create image
            plot_image = self.plot_walk_diagram(path, n_episode)

            # plot image
            fig = plt.figure()
            ax = fig.add_subplot(111, frameon=False)
            ax.imshow(plot_image)

            # save image
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig('{}diagram{:05d}.png'.format(path, n_episode))

        elif mode == 'animation':

            self.image_array.append(self.plot_walk_diagram(path, n_episode))
            
            fig = plt.figure(figsize=(30,30))
            imgs = []
            for im in self.image_array:
                plot = plt.imshow(im)
                imgs.append([plot])

            ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True)

            ani.save('{}diagram{:05d}.gif'.format(path, n_episode))

        elif mode == 'diagram':

            diagram = np.array(self.diagram_log)

            np.save(arr=diagram, file='{}numpy_diagram{:05d}.npy'.format(path, n_episode))

    def close(self):
        pass

    def _log_diagram(self):
        
        log_element = []

        log_element.append(sum(self.status_viscek))
        log_element.append(sum(self.status_catched))
        log_element.append(sum(self.status_exiting))
        log_element.append(sum(self.status_evacuated))

        self.diagram_log.append(log_element)

    def _grad_potential_pedestrians(self, alpha):
        
        r = np.expand_dims(self.agent_position, axis=0) - self.pedestrians_position
        r = r[self.status_viscek]
        
        if len(r) != 0:
            grad = - alpha / (np.linalg.norm(r, axis = 1)[:, np.newaxis] + self.eps) ** (alpha + 2) * r
            grad = grad.sum(axis = 0)
        else:
            grad = np.zeros(2)

        return grad

    def _grad_potential_exit(self, alpha):

        r = self.agent_position - self.exit

        grad = - alpha / (np.linalg.norm(r) + self.eps) ** (alpha + 2) * r
        grad *= sum(self.status_catched)

        return grad

    def _return_state(self):
        state = np.zeros(6, dtype=np.float32)

        state[0] = self.agent_position[0]
        state[1] = self.agent_position[1]
        
        state[2:4] = self._grad_potential_pedestrians(alpha = 2)
        state[4:6] = self._grad_potential_exit(alpha = 2)
        # state[6:8] = self._grad_potential_pedestrians(alpha = 3)
        # state[8:10] = self._grad_potential_exit(alpha = 3)
        # state[10:12] = self._grad_potential_pedestrians(alpha = 4)
        # state[12:] = self._grad_potential_exit(alpha = 4)
        return state

    def _return_positions(self):
        positions = np.zeros((self.N_pedestrians + 1, 2), dtype=np.float32)
        positions[0] = self.agent_position  # 2 variables
        positions[1:] = self.pedestrians_position
        return positions

    def _place(self):
        # leader
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_direction = np.zeros(2, dtype=np.float32)
        # exit
        self.exit = np.array([0.0, -1.0], dtype=np.float32)
        # pedestrians
        self.pedestrians_position = np.random.uniform(-1.0 * self.w, 1.0 * self.w, size=(self.N_pedestrians, 2))
        self.pedestrians_direction = np.zeros((self.N_pedestrians, 2), dtype=np.float32)

    def _sum_distance(self):

        distances = distance_matrix(self.pedestrians_position, np.expand_dims(self.exit, axis=0), 2)

        return distances.sum() / self.N_pedestrians

    def _create_baseline(self):

        arr_of_actions = []
        pos = np.array([0, 0], dtype='float64')

        print("RIGHT")
        while True:
            # Right
            print(pos)
            new_action = np.array([1, 0], dtype='float64')
            dir = new_action * self.step_size
            # print(pos, new_action, pos + dir)
            if self._is_wall_collision(pos + dir):
                break
            pos += dir
            arr_of_actions.append(new_action)
        
        print("UP")
        while True:
            # Up
            new_action = np.array([0, 1], dtype='float64')
            dir = new_action * self.step_size
            # print(pos, new_action, pos + dir)
            if self._is_wall_collision(pos + dir):
                break
            pos += dir
            arr_of_actions.append(new_action)
        
        new_action = np.array([-1, 0], dtype='float64')
        right_turn = np.array([1, 0], dtype='float64')
        left_turn = np.array([-1, 0], dtype='float64')
        down_action = np.array([0, -1], dtype='float64')
        dir_down = down_action * self.step_size

        print("DOWN WAY", pos)
        while True:
            # Left..Left Down Rigth...Right Down -->
            dir = new_action * self.step_size
            if self._is_wall_collision(pos + dir):
                if self._is_wall_collision(pos + dir_down):
                    break
                else:
                    # # print(pos, down_action, pos + dir_down)
                    # pos += dir_down
                    # arr_of_actions.append(down_action)
                    # pos += dir_down
                    # arr_of_actions.append(down_action)
                    # pos += dir_down
                    # arr_of_actions.append(down_action)
                    pos += dir_down
                    arr_of_actions.append(down_action)
                    if new_action[0] == left_turn[0]:
                        new_action = right_turn.copy()
                    else:
                        new_action = left_turn.copy()
            else:
                # print(pos, new_action, pos + dir)
                pos += dir
                arr_of_actions.append(new_action)
        
        # print(len(arr_of_actions))

        while(len(arr_of_actions) <= self.tmax):
            new_action = np.array([0, 0], dtype='float64')
            arr_of_actions.append(new_action)

        # print(len(arr_of_actions), self.tmax)

        return iter(arr_of_actions)
        

    def _move(self, action):

        # Receiving action
        # action = np.array(action)
        action = next(self.baseline)
        # print(action)
        np.clip(action, -1, 1, out=action)

        # Performing step of leader particle
        self.agent_direction = self.step_size * action
        # self.agent_direction = (np.random.rand(2) - 0.5) * 2 * self.step_size

        if not self._is_wall_collision(self.agent_position + self.agent_direction):
            self.agent_position += self.agent_direction
        else:
            self.reward -= 5
            return

        # Performing movement of pedestrians
        # Evacuated zone
        self.pedestrians_position[self.status_evacuated] = self.exit

        # Exiting zone
        if sum(self.status_exiting) > 0:
            vec2exit = self.exit - self.pedestrians_position[self.status_exiting]
            len2exit = np.linalg.norm(vec2exit, axis=1)
            vec_size = np.minimum(len2exit, self.step_size)
            vec2exit = (vec2exit.T / len2exit * vec_size).T

            self.pedestrians_position[self.status_exiting] += vec2exit

        # Viscek and catch zone
        self.pedestrians_direction[self.status_catched] = self.agent_direction # update directions of caught

        ######
        # v_pedestrians = self.pedestrians_direction[self.status_viscek]
        # v_staying = np.where(v_pedestrians[:, 0]**2 + v_pedestrians[:, 1]**2 == 0, True, False)
        # len_v_selected = len(v_pedestrians[v_staying])
        # v_pedestrians[v_staying] = (np.random.rand(len_v_selected, 2) - 0.5) * 2 * self.step_size
        # self.pedestrians_direction[self.status_viscek] = v_pedestrians
        ######

        status_vc = np.logical_or(self.status_viscek, self.status_catched) # init vc = viscek or caught pedestrians
        vc_directions = self.pedestrians_direction[status_vc]

        distance2moving = distance_matrix(self.pedestrians_position[self.status_viscek], 
                                          self.pedestrians_position[status_vc], 2) # distances between all viscek and vc pedestrians

        intersection = np.where(distance2moving < self.vision_radius, 1, 0)  # only short enought distances
        # movingORstaying = np.where(np.linalg.norm(self.pedestrians_direction[status_vc], axis = 1) != 0, 1, 0) # only moving particles will be taken under normiration
                    
        # viscek_direction_norm = np.maximum(1, (intersection * movingORstaying).sum(axis = 1))  # normiration (only moving particles)
        viscek_direction_norm = np.maximum(1, intersection.sum(axis = 1))  # normiration (only moving particles)
        viscek_direction_x = (intersection * vc_directions[:, 0]).sum(axis = 1) / viscek_direction_norm  # x_coordinates of new direction for all viscek particles
        viscek_direction_y = (intersection * vc_directions[:, 1]).sum(axis = 1) / viscek_direction_norm  # y_coordinates of new direction for all viscek particles

        solid_sum = np.concatenate((np.expand_dims(viscek_direction_x, axis = 1), 
                                    np.expand_dims(viscek_direction_y, axis = 1)), axis=1)
        randomization = (np.random.rand(sum(self.status_viscek), 2) - 0.5) * 2 * self.step_size / 10

        self.pedestrians_direction[self.status_viscek] = solid_sum + randomization
    
        self.pedestrians_position[status_vc] += self.pedestrians_direction[status_vc]

        clipped = np.clip(self.pedestrians_position, -1.0, 1.0)

        miss = self.pedestrians_position - clipped

        self.pedestrians_position -= 2*miss
        self.pedestrians_direction -= 2*miss

        self._status(reward_on=True)

 
    def _is_wall_collision(self, pt = None):

        if pt is None:
            pt = self.agent_position

        if pt[0] > self.w or pt[0] < -self.w or pt[1] > self.h or pt[1] < -self.h:
            return True

        return False
            
    def _is_distance_low(self, destination, radius):

        distances = distance_matrix(self.pedestrians_position, np.expand_dims(destination, axis=0), 2)
        
        return np.where(distances < radius, True, False).squeeze()

    def _status(self, reward_on=False):
        time_factor = 1 - self.t / (200 * self.N_pedestrians)

        # Evacuated
        new_status = self._is_distance_low(self.exit, self.evacuation_radius)
        # if reward_on:
        #     n_new = sum(new_status) - sum(self.status_evacuated)
        #     if n_new > 0:
        #         self.reward += 30 * n_new
        self.status_evacuated = new_status

        # Exiting
        new_status = self._is_distance_low(self.exit, self.exit_radius) * np.logical_not(self.status_evacuated)
        if reward_on:
            n_new = sum(new_status) - sum(self.status_exiting)
            if n_new > 0:
                self.reward += (15 + 10 * time_factor) * n_new
        self.status_exiting = new_status

        # Caught
        status_exiting_evacuated = np.logical_or(self.status_exiting, self.status_evacuated)
        new_status = self._is_distance_low(self.agent_position, self.vision_radius) * np.logical_not(status_exiting_evacuated)
        if reward_on:
            n_new = sum(new_status) - sum(self.status_catched)
            if n_new > 0:
                self.reward += (10 + 5 * time_factor) * n_new
        self.status_catched = new_status

        self.status_viscek = np.logical_not(np.logical_or(self.status_catched, status_exiting_evacuated))

        if sum(self.status_evacuated) + sum(self.status_exiting) + sum(self.status_catched) + sum(self.status_viscek) != self.N_pedestrians:
            print('ERROR!!!!!!!!')

    # def _grad_potential_pedestrians_old(self, alpha):
        
    #     r = []
    #     for pedestrian_id in range(self.N_pedestrians):
    #         if not (self.status_caught[pedestrian_id] or self.status_exiting[pedestrian_id]):
    #             # print('temp_r', r)

    #             pedestrian = self.pedestrians_position[pedestrian_id]

    #             r.append([self.agent_position[0] - pedestrian[0],
    #                       self.agent_position[1] - pedestrian[1]])

    #     r = np.array(r)
    #     if len(r) != 0:
    #         grad = - self.alpha / (np.linalg.norm(r, axis = 1)[:, np.newaxis] + self.eps) ** (self.alpha + 2) * r
    #         grad = grad.sum(axis = 0)
    #     else:
    #         grad = np.zeros(2)

    #     return grad


if __name__ == '__main__':

    env = GameEnv(NUM_PEDESTRIANS=2, VISION_RADIUS=0.2)

    obs = env.reset()
    # env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())
    env.render(mode='bg')

    n_steps = 10
    actions = [3, 2, 2, 2, 1,
               3, 3, 3, 0, 1,
               1, 2, 1, 1, 1]

    # for step, ACTION in zip(range(n_steps), actions):
    # ACTION = random.randint(0, 3)
    for step in range(n_steps):
        ACTION = (np.random.rand(2) - 0.5) * 2
        obs, reward, done, info = env.step(ACTION)
        # print("Step {} with ACTION={}".format(step + 1, ACTION))
        # print('obs=', obs, 'reward=', reward, 'done=', done)
        env.create_walk_diagram(mode='one_image')
        # if step % 5 ==0:
        #     env.render(mode = 'consolfe')
        if done:
            print("Goal reached!", "reward=", reward)
            break

    env.record(mode='one_image')
    # env.record(mode='animation') 