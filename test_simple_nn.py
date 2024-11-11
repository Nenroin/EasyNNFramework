import numpy as np

from v1.src.base.activation import linear
from v1.src.base.data.model_data_source import ModelDataSource
from v1.src.base.layers.dropout_layer import DropoutLayer
from v1.src.base.layers.input_layer import InputLayer
from v1.src.base.layers.linear_layer import LinearLayer
from v1.src.base.loss_function import mse
from v1.src.base.metrics import LossMetric
from v1.src.base.optimizers.sgd import SGD
from v1.src.base.models.custom_sequential_model import CustomSequentialModel

x_train = np.array([])
y_train = np.array([])

x_test = np.array([])
y_test = np.array([])

def function(x):
    return np.multiply(1, np.sin(np.multiply(9, x))) + 0.5

for i in range(30):
    step = 0.025
    x_train = np.append(x_train, [function(step * i), function(step * (i + 1)), function(step * (i + 2)),
                                  function(step * (i + 3))])
    y_train = np.append(y_train, function(step * (i + 4)))
x_train = x_train.reshape((-1, 4))
y_train = y_train.reshape((-1, 1))

for i in range(31, 46):
    step = 0.025
    x_test = np.append(x_train, [function(step * i), function(step * (i + 1)), function(step * (i + 2)),
                                 function(step * (i + 3))])
    y_test = np.append(y_train, function(step * (i + 4)))
x_test = x_test.reshape((-1, 4))
y_test = y_test.reshape((-1, 1))

data_source = ModelDataSource(
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    shuffle=False,
    batch_size=1,
)

model = CustomSequentialModel(layers=[
    InputLayer(4),
    DropoutLayer(4,
                 rate=0.0
                 ),
    LinearLayer(1,
                activation=linear(),
                use_bias=True,
                ),
])

model.init_layers_params()

model.build(loss_function=mse(),
            optimizer=SGD(learning_rate=0.2, nesterov=True),
            metric=LossMetric(loss_function=mse()))

model.fit(model_data_source=data_source, epochs=10, disable_tqdm=True)