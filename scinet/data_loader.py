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


def load(validation_size_p, file_name):
    """
    Params:
    validation_size_p: percentage of data to be used for validation
    file_name (str): File containing the data
    """
    f = gzip.open(io.data_path + file_name + ".plk.gz", 'rb')
    data, states, params = cPickle.load(f)
    states = np.array(states)
    train_val_separation = int(len(data[0]) * (1 - validation_size_p / 100.))
    training_data = [data[i][:train_val_separation] for i in [0, 1, 2]]
    training_states = states[:train_val_separation]
    validation_data = [data[i][train_val_separation:] for i in [0, 1, 2]]
    validation_states = states[train_val_separation:]
    f.close()
    return (training_data, validation_data, training_states, validation_states, params)
