from v1.src.base.activation import *
from v1.src.base.data import data_augmentation
from v1.src.base.layers.linear_layer import *
from v1.src.base.metrics import AccuracyMetric
from v1.src.base.metrics.matching_functon import one_hot_matching_function
from v1.src.base.models.custom_sequential_model import *
from v1.src.base.optimizers.adam import Adam
from v1.src.base.value_initializer import he_initializer
from v1.src.mnist.mnist_dataloader import MnistDataloader

mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


batch_size = 200

data_source = ModelDataSource(
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    data_augmentations=[
        data_augmentation.flatten(),
        data_augmentation.normalize(),
        data_augmentation.one_hot_labels(num_classes=10)
    ],
    batch_size=batch_size,
)

model = SequentialModel(
    layers=[
        InputLayer(784),
        LinearLayer(256,
                    activation=relu(),
                    prev_weights_initializer=he_initializer(),
                    ),
        LinearLayer(10,
                    activation=softmax(),
                    prev_weights_initializer=he_initializer(),
                    ),
    ]
)

model.init_layers_params()

model.build(loss_function=mse(),
            optimizer=Adam(learning_rate=0.0011),
            metrics=[
                AccuracyMetric(matching_function=one_hot_matching_function())
            ],
            )

model.fit(model_data_source=data_source,
          epochs=10)