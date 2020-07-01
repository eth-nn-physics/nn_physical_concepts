#   Copyright 2020 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
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
from ed_copernicus import fix_jumps_series
from ed_copernicus import fix_jumps_for_plot


def theta_mars_from_true_anomaly(true_anomaly_earth, true_anomaly_mars):
    """
    Calculate the angle of Mars with respect to a fixed star as seen from Earth
    Params:
    true_anomaly_earth: true anomaly of Earth
    true_anomaly_mars: true anomaly of Mars
    """
    AU = 149597870700
    a_earth = 1.00000011 * AU
    ecc_earth = 0.01671022
    a_mars = 1.52366231 * AU
    ecc_mars = 0.09341233
    R_earth = get_radius(true_anomaly_earth, a_earth, ecc_earth)
    R_mars = get_radius(true_anomaly_mars, a_mars, ecc_mars)
    dist = np.sqrt(R_mars**2 + R_earth**2 - 2 * R_mars * R_earth * np.cos(true_anomaly_mars - true_anomaly_earth))
    sin_theta_M = (R_earth * np.sin(true_anomaly_earth) - R_mars * np.sin(true_anomaly_mars)) / dist
    cos_theta_M = (R_earth * np.cos(true_anomaly_earth) - R_mars * np.cos(true_anomaly_mars)) / dist
    return np.angle(cos_theta_M + 1.j * sin_theta_M)


def copernicus_data(series_length, N=None, delta_t=25, file_name=None, t_earth_target=None, t_mars_target=None):
    """
    Params:
    series_length: number of samples in each time series
    N: number of training examples
    delta_t: number of days between adjacent observations in a time series
    file_name: if given, the data is saved to a file of this name
    t_earth_target, t_mars_target: used for plotting, do not use for data generation
    """
    delta_t = float(delta_t)
    T_earth = 365.256
    ecc_earth = 0.01671022
    T_mars = 686.97959
    ecc_mars = 0.09341233

    if t_earth_target is None and t_mars_target is None:
        t_0_E = T_earth * np.random.rand(N)
        t_0_M = T_mars * np.random.rand(N)
    else:
        N = len(t_earth_target)
        t_0_E = np.array(t_earth_target)
        t_0_M = np.array(t_mars_target)

    phi_E = np.vstack([get_true_anomaly(delta_t * s + t_0_E, T_earth, ecc_earth) for s in range(series_length)]).T
    phi_M = np.vstack([get_true_anomaly(delta_t * s + t_0_M, T_mars, ecc_mars) for s in range(series_length)]).T
    phi_E = fix_jumps_series(phi_E)
    phi_M = fix_jumps_series(phi_M)
    theta_S = phi_E
    theta_M = theta_mars_from_true_anomaly(phi_E, phi_M)
    theta_M = fix_jumps_series(theta_M)

    if t_earth_target is not None:
        theta_M = fix_jumps_for_plot(theta_M)

    data = np.dstack([theta_S, theta_M])
    data = np.reshape(data, [N, 2 * series_length], order='C')

    states = np.dstack([phi_E, phi_M])

    result = (data, states)

    if file_name is not None:
        f = gzip.open(io.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result


def get_true_anomaly(t, period, eccen):
    """
    Params:
    t = time since passing the perihelion
    period = orbital period
    eccen = eccentric
    """
    meananom = mean_anom(t, period)
    eccentric_anomaly = ecc_anom(eccen, meananom)
    trueanomaly = true_anom(eccen, eccentric_anomaly)
    return trueanomaly


def get_radius(trueanomaly, a, eccen):
    """
    Params:
    trueanomaly = true anomaly
    a =  semi-major axis
    eccen = eccentric
    """
    r = a * (1.0 - eccen ** 2) / (1.0 + (eccen * np.cos(trueanomaly)))
    return r

def ecc_anom(ec, am, dp=14, max_iter=1000):
    """
    Params:
    ec = eccentricity;
    am = mean anomaly
    dp = number of decimal places (precision for calculation)
    max_iter = maximal number of iteration for approximating the eccentric anomaly (the equation
               cannot be solved analytically)
    """
    i = 0
    delta = np.power(10., -dp)
    E = am
    F = E - ec*np.sin(E) - am
    # Credit: https://stackoverflow.com/questions/5287814/solving-keplers-equation-computationally
    while np.all(np.abs(F) > delta) and i < max_iter:
        E = E - F/(1.0-(ec * np.cos(E)))
        F = E - ec * np.sin(E) - am
        i = i + 1
    if i == max_iter:
        indices = np.where(np.abs(F) > delta)
        print("Warning: Simulated orbit might not be accurate enough for mean anomaly = ", am[indices])
    return np.round(E*np.power(10., dp))/np.power(10., dp)


def true_anom(ec, e):
    """
    Params:
    ec = eccentricity
    e = eccentric anomaly
    """
    phi = 2.0 * np.arctan(np.sqrt((1.0 + ec)/(1.0 - ec)) * np.tan(e / 2.0))
    return phi


def mean_anom(time, period):
    """
    Params:
    time = time elapsed since passing the perihelion
    period = orbital period
    """
    n = 2 * np.pi / period
    m = n * time
    return m
