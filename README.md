# sport_predicting_ffnn
___

#### Project's modules:
 * **FFNN** - a module, which contains class of a feed forward neural network. The n_network is a reproduced and reworked version of a network shown there - (source: https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part3_neural_network_mnist_data_with_rotations.ipynb). The changes made the network more universal because now it allows to create any amount of its layers and nodes in each layer.
 * **weights_recorder** - a module, which allows to init, learn and train the FFNN to save weights of the network, in order reuse them later.
 * **opt_live_predictor** - its main purpose is to parse a sport statistics web-site and return a list of data, required for predicting a sport event results (contains only a template of how it might look).
 * **gui** - a module, which contains several classes, that are responsible for predicting sport events and visualization of the predictions.

#### UI-cases:

For running all possible states of the program interface:
1) Comment out line 26 in the gui.py module:
![comment](https://user-images.githubusercontent.com/51992590/90664762-c005f400-e253-11ea-8fe1-5f3cfe552861.PNG)
2) Uncomment any out of 5 commented test cases placed in lines from 29th to 54th in the gui.py module:
![case](https://user-images.githubusercontent.com/51992590/90665138-51756600-e254-11ea-9429-95c9d2183b88.PNG)
3) Run gui.py.

If you made everything right, you would be able to see similar to on of the next two screenshots:
![gui_cases](https://user-images.githubusercontent.com/51992590/90663920-f55e1200-e252-11ea-934d-5d7f672381e3.png)
