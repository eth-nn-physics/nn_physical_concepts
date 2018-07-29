#   Copyright 2018 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import cPickle
import gzip
import io
import numpy as np

# Create data for collision of rotating mass (can only move on a circle) and a free flying mass. Here we consider constant masses (m_1,m_2=1) and a constant radius (r=1) of the orbit.
# int N=number of samples
# int N_inp1=number of time steps for input into Encoder
# int N_inp2=number of time steps for input into Decoder

# Note: for training a network, we have 6*N_inp1 neurons for the first input, and 3*N_inp2+1 neurons for the second input


def create_data(N, N_inp1, N_inp2, fileName=None):
    # Initialize variables
    input1 = np.empty([N, 6 * N_inp1])  # input time series for (x,y,t) of rotating mass and free particle
    # input time series for (x,y,t) for the positions of the free particle after the interaction, and the time t_p at which we want to predict
    input2 = np.empty([N, 3 * N_inp2 + 1])
    output = np.empty([N, 2])  # predicted (x,y) at time t_p
    # Choose physical settings before collision
    vx_in = gen_random_array(N, [1, 2])  # lower bound must be strictly bigger than zero
    vy_in = gen_random_array(N, [-1, 1])
    w_in = gen_random_array(N, [1, 3])
    x0_in = gen_random_array(N, [-3, -1])
    # Determine other variables such that a collision will occur
    t_col = -x0_in / vx_in
    y0_in = calculate_y0(t_col, vy_in)  # y coordinate of starting point of the free particle (note that the collision should occur at the point (0,1)
    al0 = calculate_al0(t_col, w_in)  # starting angle alpha_0 for the rotating particle (we measure the angle w.r.t. the x-axis)
    # Choose the physical setting after the collision
    vx_out = gen_random_array(N, [-2, 2])  # velocity in x-direction of outgoing particle
    vy_out = gen_random_array(N, [-2, 2])  # velocity in y-direction of outgoing particle
    time_steps_out_rot = gen_random_array(N, [0, 4])  # Choose times at random at which we want to predict where the rotating particle is
    # Create the input data to train (and test) the NN
    for i in range(N):
        t_collision = -x0_in[i] / vx_in[i]
        time_steps_in_free = generate_random_time_steps(0., t_collision, N_inp1)
        time_steps_in_rot = generate_random_time_steps(0., t_collision, N_inp1)
        time_steps_out_free = generate_random_time_steps(0., 4., N_inp2)
        for j in range(N_inp1):
            input1[i, 0 * N_inp1 + j] = gen_input1_x(x0_in[i], vx_in[i], time_steps_in_free[j]) + np.random.normal(0, 0.01, 1)  # Adding some noise
            input1[i, 1 * N_inp1 + j] = gen_input1_y(y0_in[i], vy_in[i], time_steps_in_free[j]) + np.random.normal(0, 0.01, 1)
            input1[i, 2 * N_inp1 + j] = time_steps_in_free[j]
            input1[i, 3 * N_inp1 + j] = gen_input1_rot_x(al0[i], w_in[i], time_steps_in_rot[j]) + np.random.normal(0, 0.01, 1)
            input1[i, 4 * N_inp1 + j] = gen_input1_rot_y(al0[i], w_in[i], time_steps_in_rot[j]) + np.random.normal(0, 0.01, 1)
            input1[i, 5 * N_inp1 + j] = time_steps_in_rot[j]
        for j in range(N_inp2):
            # Note that the time measurement for gen_inpu2_x starts at the collision
            input2[i, 0 * N_inp2 + j] = gen_input2_x(vx_out[i], time_steps_out_free[j]) + np.random.normal(0, 0.01, 1)
            input2[i, 1 * N_inp2 + j] = gen_input2_y(vy_out[i], time_steps_out_free[j]) + np.random.normal(0, 0.01, 1)
            input2[i, 2 * N_inp2 + j] = time_steps_out_free[j]
        input2[i, 3 * N_inp2] = time_steps_out_rot[i]
    # Create output data
    for i in range(N):
        output[i, 0:2] = np.reshape(generate_output(time_steps_out_rot[i], vx_out[i], vx_in[i], w_in[i]), [-1, 2]) + np.random.normal(0, 0.01, [1, 2])
    result = ([input1, input2, output], np.transpose(np.array([x0_in, y0_in, vx_in, vy_in, al0, w_in, vx_out, vy_out])), [])
    if fileName is not None:
        f = gzip.open(io.data_path + fileName + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return (result)


# Helper methods

def gen_random_array(N, interval):
    return (interval[1] - interval[0]) * np.random.rand(N) + interval[0]


def calculate_al0(t_col, w_in):
    return -t_col * w_in + np.pi / 2


def calculate_y0(t_col, vy_in):
    return -t_col * vy_in + 1


def generate_random_time_steps(start_time, end_time, step_num):
    steps_pure = np.linspace(start_time, end_time, step_num)
    step_size = steps_pure[1] - steps_pure[0]
    steps_noisy = steps_pure + np.random.uniform(-step_size / 2., step_size / 2., steps_pure.shape)
    steps_noisy[0] = 0.
    return steps_noisy


def gen_input1_x(x0, vx, time):
    return x0 + vx * time


def gen_input1_y(y0, vy, time):
    return y0 + vy * time


def calculte_angle(al0, w, time):
    return al0 + w * time


def gen_input1_rot_x(al0, w, time):
    angle = calculte_angle(al0, w, time)
    return np.cos(angle)


def gen_input1_rot_y(al0, w, time):
    angle = calculte_angle(al0, w, time)
    return np.sin(angle)


def gen_input2_x(vx_out, time):
    return vx_out * time


def gen_input2_y(vy_out, time):
    return 1. + vy_out * time


def total_ang_mom(vx, w):
    return w - vx


def predicted_angle(time, vx_out, ang_mom):
    w_prime = ang_mom + vx_out  # Here we use that the total angular momentum is conserved
    return np.pi / 2. + time * w_prime


def generate_output(time, vx_out, vx_in, w_in):
    ang_mom = total_ang_mom(vx_in, w_in)
    angle_pred = predicted_angle(time, vx_out, ang_mom)
    return [np.cos(angle_pred), np.sin(angle_pred)]
