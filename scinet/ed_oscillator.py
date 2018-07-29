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


def osc_eqn(A_0, delta_0, b, kappa, t):
    return np.real(A_0 * np.exp(-b / 2. * t) * np.exp(1 / 2. * np.sqrt(b**2 - 4 * kappa + 0.j) * t + 1.j * delta_0))


def oscillator_data(N, t_sample=np.linspace(0, 5, 50), A_interval=[1, 1], delta_interval=[0, 0],
                    b_interval=[0.5, 1], kappa_interval=[5, 10], t_meas_interval=None, fileName=None):
    t_sample = np.array(t_sample, dtype=float)
    # cover edges
    b_interval = [0.8 * b_interval[0], 1.2 * b_interval[1]]
    kappa_interval = [0.8 * kappa_interval[0], 1.2 * kappa_interval[1]]
    bb = (b_interval[1] - b_interval[0]) * np.random.rand(N) + b_interval[0]
    kk = (kappa_interval[1] - kappa_interval[0]) * np.random.rand(N) + kappa_interval[0]
    AA = (A_interval[1] - A_interval[0]) * np.random.rand(N) + A_interval[0]
    dd = (delta_interval[1] - delta_interval[0]) * np.random.rand(N) + delta_interval[0]
    x_in = []
    if t_meas_interval is None:
        t_meas_interval = [t_sample[0], 2 * t_sample[-1]]
    t_meas = np.reshape(np.random.rand(N) * (t_meas_interval[1] - t_meas_interval[0]) + t_meas_interval[0], [N, 1])
    x_out = []
    for b, kappa, A_0, delta_0, t in zip(bb, kk, AA, dd, t_meas):
        x_in.append(osc_eqn(A_0, delta_0, b, kappa, t_sample))
        x_out.append(osc_eqn(A_0, delta_0, b, kappa, t))
    x_in = np.array(x_in)
    x_out = np.reshape(x_out, [N, 1])
    state_list = np.vstack([AA, dd, bb, kk]).T
    result = ([x_in, t_meas, x_out], state_list, [])
    if fileName is not None:
        f = gzip.open(io.data_path + fileName + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return (result)
