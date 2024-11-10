import cProfile
import pstats
from os.path import join


from v1.src.base.activation import *
from v1.src.base.data import data_augmentation
from v1.src.base.data.model_data_source import ModelDataSource
from v1.src.base.layers.input_layer import InputLayer
from v1.src.base.layers.linear_layer import *
from v1.src.base.loss_function import mse
from v1.src.base.metrics.accuracy_metric import one_hot_matches_metric
from v1.src.base.models.custom_sequential_model import *
from v1.src.base.optimizers.adam import Adam
from v1.src.base.value_initializer import he_initializer
from v1.src.mnist.mnist_dataloader import MnistDataloader

mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# pr = cProfile.Profile()
# pr.enable()

batch_size = 200

data_source = ModelDataSource(
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    train_data_augmentations=[
        # data_augmentation.scaling_pil(
        #     copies=1
        # ),
        # data_augmentation.cropping(
        #     copies=1
        # ),
    ],
    data_augmentations=[
        data_augmentation.flatten(),
        data_augmentation.normalize(),
        data_augmentation.one_hot_labels(num_classes=10)
    ],
    shuffle=True,
    batch_size=batch_size,
)

model = SequentialModel(
    layers=[
        InputLayer(784),
        LinearLayer(256,
                    activation=relu(),
                    prev_weights_initializer=he_initializer(),
                    ),
        LinearLayer(50,
                    activation=relu(),
                    prev_weights_initializer=he_initializer(),
                    ),
        LinearLayer(10,
                    activation=linear(),
                    prev_weights_initializer=he_initializer(),
                    ),
    ]
)

model.init_layers_params()

print(f"learning_rate: {0.002}")

model.build(loss_function=mse(),
            # optimizer=SGD(learning_rate=0.0035),
            optimizer=Adam(learning_rate=0.002),
            # optimizer=AdaGrad(learning_rate=0.0185),
            metric=one_hot_matches_metric())

model.fit(model_data_source=data_source,
          epochs=10)

# pr.disable()
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.dump_stats("profiler_files/profile_nn_10_epochs.prof")
