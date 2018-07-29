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

import numpy as np
import cPickle
import gzip
import io as io_paths
from scipy.stats import unitary_group


def random_state(qubit_num):
    return unitary_group.rvs(2**qubit_num)[:, 0]


def random_subspace_states(qubit_num, k, states_num):
    """
    qubit_num: number of qubits
    k: number of orthogonal basis vectors
    states_num: number of states randomly sampled from subspace
    """

    assert(2 * 2**qubit_num > k)
    output_states = []
    subspace_basis = (unitary_group.rvs(2**qubit_num)[:, :k]).T
    for _ in range(states_num):
        c = np.random.rand(k) - 0.5
        linear_combination = 0.j
        for i in range(k):
            linear_combination += c[i] * subspace_basis[i]
        output_states.append(linear_combination / np.linalg.norm(linear_combination))
    return output_states


def projection(a, b):
    return np.abs(np.dot(np.conj(a), b))**2


def create_data(qubit_num, measurement_num1, measurement_num2, sample_num, file_name=None, incomplete_tomography=[False, False]):
    """
    Params:
    qubit_num: number of qubits
    measurement_num1: number of projective measurements to be performed on input qubit
    measurement_num2: number of projective measurements to be performed on projection axis
    sample_num: number of training examples to be generated
    file_name: file is stored in /data/file_name.pkl.gz
    incomplete_tomography: if the i-th entry is k, then the states for the projectors M_i are sampled from a k-dimensional real subspace
    """
    states_in1 = np.empty([sample_num, 2**qubit_num], dtype=np.complex_)
    states_in2 = np.empty([sample_num, 2**qubit_num], dtype=np.complex_)
    meas_res1 = np.empty([sample_num, measurement_num1], dtype=np.float_)
    meas_res2 = np.empty([sample_num, measurement_num2], dtype=np.float_)
    output = np.empty([sample_num, 1])
    if incomplete_tomography[0]:
        fixed_states_in1 = random_subspace_states(qubit_num, incomplete_tomography[0], measurement_num1)
    else:
        fixed_states_in1 = [random_state(qubit_num) for _ in range(measurement_num1)]
    if incomplete_tomography[1]:
        fixed_states_in2 = random_subspace_states(qubit_num, incomplete_tomography[1], measurement_num2)
    else:
        fixed_states_in2 = [random_state(qubit_num) for _ in range(measurement_num2)]
    for i in range(sample_num):
        states_in1[i] = random_state(qubit_num)
        states_in2[i] = random_state(qubit_num)
        meas_res1[i] = np.array([projection(s1, states_in1[i]) for s1 in fixed_states_in1])
        meas_res2[i] = np.array([projection(s2, states_in2[i]) for s2 in fixed_states_in2])
        output[i, 0] = projection(states_in1[i], states_in2[i])
    result = ([meas_res1, meas_res2, output], [states_in1, states_in2], [fixed_states_in1, fixed_states_in2])
    if file_name is not None:
        f = gzip.open(io_paths.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result
