from v1.src.base.activation import *
from v1.src.base.data import data_augmentation
from v1.src.base.data.model_data_source import ModelDataSource
from v1.src.base.layers.input_layer import InputLayer
from v1.src.base.layers.linear_layer import *
from v1.src.base.loss_function import mse
from v1.src.base.metrics.accuracy_metric import one_hot_matches_metric
from v1.src.base.models.custom_sequential_model import *
from v1.src.base.models.model import Model
from v1.src.base.optimizers.adam import Adam
from v1.src.base.value_initializer import he_initializer
from v1.src.mnist.mnist_dataloader import MnistDataloader

mnist_dataloader = MnistDataloader(
    training_images_filepath = "../v1/src/mnist/data/train-images-idx3-ubyte/train-images-idx3-ubyte",
    training_labels_filepath = "../v1/src/mnist/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
    test_images_filepath = "../v1/src/mnist/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    test_labels_filepath = "../v1/src/mnist/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

batch_size = 200

data_source = ModelDataSource(
    test_data=(x_test, y_test),
    data_augmentations=[
        data_augmentation.flatten(),
        data_augmentation.normalize(),
        data_augmentation.one_hot_labels(num_classes=10)
    ],
    batch_size=batch_size,
)

model = Model.load_from_file("model_Adam_lr0.0011.txt")

model.test_epoch(test_data=data_source.test_data_batches(1))
