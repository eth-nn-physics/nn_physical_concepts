# Discovering physical concepts with neural networks

Code for: R. Iten, T. Metger, H.Wilming, L. del Rio, and R. Renner. "Discovering physical concepts with neural networks",[	arXiv:1807.10300](https://arxiv.org/abs/1807.10300) (2018).

This repository contains the trained [Tensorflow](https://www.tensorflow.org) models used in the paper as well as code to load, train and analyze them.

An overview of how this work relates to other research on the use of AI for the discovery of physical concepts, and recent advances based on this research, are presented in the book ["Artificial Intelligence for Scientific Discoveries"](https://www.amazon.de/Artificial-Intelligence-Scientific-Discoveries-Experimental/dp/3031270185/ref=sr_1_3?__mk_de_DE=ÅMÅŽÕÑ&crid=SZL7B1KF8C4O&keywords=Artificial+Intelligence+for+Scientific+Discoveries&qid=1675959397&sprefix=artificial+intelligence+for+scientific+discoveries%2Caps%2C64&sr=8-3) (2023).

Requires:

- Python 2.7
- ``numpy``
- ``matplotlib``
- ``tensorflow``
- ``tensorboard``
- ``tqdm``
- ``jupyter``

Branches:

- ``master``: Implementation of [beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl) [1] for reference. Includes an example in the ``/analysis`` folder that shows how to set up and train a network.
- ``pendulum``: *SciNet* finds correct physical parameters describing a damped pendulum.
- ``angular_momentum``: *SciNet* finds and exploits angular momentum conservation to make predictions.
- ``qubit``: *SciNet* recovers correct number of parameters describing quantum states.
- ``copernicus``: *SciNet* discovers heliocentric model of the solar system.

To use the code:

1. Clone the repository.
2. Add the cloned directory ``nn_physical_concepts`` to your python path. See [here](https://stackoverflow.com/questions/10738919/how-do-i-add-a-path-to-pythonpath-in-virtualenv) for instructions for doing this in a virtual environment. Without a virtual environment, see [here](https://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath).
3. Import `from scinet import *`. This includes the shortcuts `nn` to the `model.py` code and `dl` to the `data_loader.py` code.
4. Import additional files (e.g. data generation scripts) using `import scinet.my_data_generator as my_data_gen_name`.

Generated data files are stored in the ``data`` directory. Saved models are stored in the ``tf_save`` directory. Tensorboard logs are stored in the ``tf_log`` directory.

Some documentation is available in the code. For further questions, please contact us directly.

[1]  Higgins, I. *et al.* beta-VAE: "Learning Basic Visual Concepts with a Constrained Variational Framework", *ICLR* (2017).

