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


def fix_jumps_series(theta_M):
    # This fixes jumps (i.e. where theta goes from pi to -pi). There may be more comprehensible ways to do this, but this works.
    while True:
        diff = np.abs(theta_M[:, 1:] - theta_M[:, :-1])
        jumps = np.array(np.where(diff > 5.)).T
        if len(jumps) == 0:
            break
        fixed_lines = []
        for x, y in jumps:
            if x in fixed_lines:
                continue
            else:
                fixed_lines.append(x)
            theta_M[x, y + 1:] = theta_M[x, y + 1:] - np.sign(theta_M[x, y + 1] - theta_M[x, y]) * 2 * np.pi
    return theta_M


def fix_jumps_for_plot(theta_M):
    # Step 1: make theta_M go from -pi to pi (approximately)
    pi_offset = np.rint((theta_M[0, 0] + np.pi / 2.) / (2 * np.pi))
    theta_M = theta_M - 2 * np.pi * pi_offset
    # Step 2: Fix jumps
    while True:
        diff = np.abs(theta_M[1:, 0] - theta_M[:-1, 0])
        jumps = np.ravel(np.array(np.where(diff > 5.)).T)
        if len(jumps) == 0:
            break
        theta_M[jumps[0] + 1:, :] = theta_M[jumps[0] + 1:, :] + np.sign(theta_M[jumps[0], 0] - theta_M[jumps[0] + 1, 0]) * 2 * np.pi
    return theta_M


def theta_M_from_phi(phi_S, phi_M):
    R_S = 1
    R_M = 1.52366231
    dist = np.sqrt(R_M**2 + R_S**2 - 2 * R_M * R_S * np.cos(phi_M - phi_S))

    sin_theta_M = (R_S * np.sin(phi_S) - R_M * np.sin(phi_M)) / dist
    cos_theta_M = (R_S * np.cos(phi_S) - R_M * np.cos(phi_M)) / dist
    return np.angle(cos_theta_M + 1.j * sin_theta_M)


def copernicus_data(series_length, N=None, delta_t=7, file_name=None, phi_S_target=None, phi_M_target=None, random_start=True):
    """
    Params:
    series_length: number of samples in each time series
    N: number of training examples
    delta_t: number of days between adjacent observations in a time series
    file_name: if given, the data is saved to a file of this name
    phi_S_target, phi_M_target: used for plotting, do not use for data generation
    random_start: if True, the starting positions for each time series are chosen randomly. If False, the starting positions are selected from the set of observations Copernicus could have made during his lifetime.
    """

    delta_t = float(delta_t)

    T_earth = 365.26
    T_mars = 686.97959

    if phi_S_target is None and phi_M_target is None:
        if random_start:
            phi_0_S = 2.4 * np.pi * np.random.rand(N) - 0.2 * np.pi
            phi_0_M = 2 * np.pi * np.random.rand(N)
        else:
            day_num = 25657  # number of days Copernicus lived
            # This is the the starting position (mean anomaly) on Copernicus# birth
            phi_S_possible = 0.965 + 2 * np.pi / T_earth * delta_t * np.arange(int(day_num / delta_t))
            phi_M_possible = 5.938 + 2 * np.pi / T_mars * delta_t * np.arange(int(day_num / delta_t))
            phi_0_S = np.mod(phi_S_possible[((len(phi_S_possible) - 1) * np.random.rand(N)).astype(int)], 2 * np.pi)
            phi_0_M = np.mod(phi_S_possible[((len(phi_M_possible) - 1) * np.random.rand(N)).astype(int)], 2 * np.pi)

    else:
        N = len(phi_S_target)
        phi_0_S = np.array(phi_S_target)
        phi_0_M = np.array(phi_M_target)
    phi_S = np.vstack([phi_0_S + 2 * np.pi / 365. * delta_t * s for s in range(series_length)]).T
    phi_M = np.vstack([phi_0_M + 2 * np.pi / T_mars * delta_t * s for s in range(series_length)]).T

    theta_S = phi_S
    theta_M = theta_M_from_phi(phi_S, phi_M)
    theta_M = fix_jumps_series(theta_M)

    if phi_S_target is not None:
        theta_M = fix_jumps_for_plot(theta_M)

    data = np.dstack([theta_S, theta_M])
    data = np.reshape(data, [N, 2 * series_length], order='C')

    states = np.dstack([phi_S, phi_M])

    result = (data, states)

    if file_name is not None:
        f = gzip.open(io.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result
