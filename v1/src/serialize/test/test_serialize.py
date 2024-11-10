import pickle

from v1.src.base.activation import relu, linear
from v1.src.base.layers.dropout_layer import DropoutLayer
from v1.src.base.layers.input_layer import InputLayer
from v1.src.base.layers.linear_layer import LinearLayer
from v1.src.base.loss_function import mse
from v1.src.base.metrics.accuracy_metric import one_hot_matches_metric
from v1.src.base.models.custom_sequential_model import CustomSequentialModel
from v1.src.base.optimizers.sgd import SGD
from v1.src.base.value_initializer import he_initializer

model = CustomSequentialModel(
    layers=[
        InputLayer(784),
        LinearLayer(256,
                    activation=relu(),
                    prev_weights_initializer=he_initializer(),
                    ),
        DropoutLayer(256,
                     rate=0.65,
                     ),
        LinearLayer(10,
                    activation=linear(),
                    prev_weights_initializer=he_initializer(),
                    ),
    ],
    metric=one_hot_matches_metric(),
    optimizer=SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
    loss_function=mse()
)

model.init_layers_params()

deser_model = None

with open("test.model", "wb") as fp:
    pickle.dump(model, fp)

with open("test.model", "rb") as fp:
    deser_model = pickle.load(fp)

deser_model.summary()