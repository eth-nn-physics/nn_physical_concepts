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
import io


def create_data(qubit_num, measurement_num, sample_num, file_name=None):
    measurements = np.empty([sample_num, measurement_num], dtype=np.float_)
    states = np.empty([sample_num, 2**qubit_num], dtype=np.complex_)
    projectors = [random_state(qubit_num) for _ in range(measurement_num)]
    for i in range(sample_num):
        sample = random_state(qubit_num)
        states[i] = sample
        measurements[i] = np.array([projection(p, sample) for p in projectors])
    result = (measurements, states, projectors)
    if file_name is not None:
        f = gzip.open(io.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result


def random_state(qubit_num):
    real = np.random.rand(2**qubit_num) - 0.5
    im = np.random.rand(2**qubit_num) - 0.5
    state = real + 1.j * im
    return state / np.linalg.norm(state)


def projection(a, b):
    return np.abs(np.dot(np.conj(a), b))**2
