# sport_predicting_ffnn
___

#### Project's modules:
 * **FFNN** - a module, which contains class of a feed forward neural network. The n_network is a reproduced and reworked version of a network shown there - (source: https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part3_neural_network_mnist_data_with_rotations.ipynb). The changes made the network more universal because now it allows to create any amount of its layers and nodes in each layer.
 * **weights_recorder** - a module, which allows to init, learn and train the FFNN to save weights of the network, in order reuse them later.
 * **opt_live_predictor** - its main purpose is to parse a sport statistics web-site and return a list of data, required for predicting a sport event results (contains only a template of how it might look).
 * **gui** - a module, which contains several classes, that are responsible for predicting sport events and visualization of the predictions.
