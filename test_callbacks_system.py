from v1.src.base.activation import linear
from v1.src.base.callbacks.default_callbacks.print_callback import PrintCallback
from v1.src.base.data import ModelDataSource
from v1.src.base.layers import InputLayer, LinearLayer
from v1.src.base.loss_function import mse
from v1.src.base.metrics import LossMetric
from v1.src.base.models import SequentialModel
from v1.src.base.optimizers import SGD

x_train = x_test = [
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6],
    [4,5,6,7]
]

y_train = y_test = [
    [1],
    [2],
    [3],
    [4]
]

data_source = ModelDataSource(
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    shuffle=False,
    batch_size=1,
)

model = SequentialModel(layers=[
    InputLayer(4),
    LinearLayer(1,
                activation=linear(),
                use_bias=True,
                ),
])

model.init_layers_params()

model.build(loss_function=mse(),
            optimizer=SGD(learning_rate=0.2, nesterov=True),
            metric=LossMetric(loss_function=mse()))

model.fit(
    model_data_source=data_source,
    epochs=10,
    callbacks=[
        PrintCallback(),
    ],
)