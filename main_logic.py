from src.base.activation import *
from src.base.callbacks.default_callbacks import ProgressBarCallback, ModelSaveCallback
from src.base.data import ModelDataSource
from src.base.data import data_augmentation
from src.base.layers import InputLayer, LinearLayer
from src.base.loss_function import mse
from src.base.metrics import AccuracyMetric, one_hot_matching_function, LossMetric
from src.base.models import SequentialModel
from src.base.optimizers.adam import Adam
from src.base.value_initializer import he_initializer
from src.mnist.mnist_dataloader import MnistDataloader
from src.base.models import Model
import matplotlib.pyplot as plt

mnist_dataloader = MnistDataloader(
    training_images_filepath="src/mnist/data/train-images-idx3-ubyte/train-images-idx3-ubyte",
    training_labels_filepath="src/mnist/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
    test_images_filepath="src/mnist/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    test_labels_filepath="src/mnist/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

batch_size = 300
epochs = 10
learning_rate = 0.001

data_source = ModelDataSource(
    test_data=(x_test[:100], y_test[:100]),
    data_augmentations=[
        data_augmentation.scaling_pil(
            scale_range=(0.5, 1.5)
        ),
        data_augmentation.rotating(
            rotation_range=(-60, 60),
            padding_value=0),
    ],
    shuffle=False,
    batch_size=100,
)


def show_images(images, title_texts):
    cols = 10
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(25, 25))
    index = 1
    for x, _ in zip(images, title_texts):
        image = x
        plt.subplot(rows, cols, index)
        plt.imshow(image)
        index += 1

    plt.show()


first_batch = data_source.test_data_batches()[0]
x_batch, y_bach = first_batch

show_images(x_batch, y_bach)

data_source = ModelDataSource(
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    test_data_augmentations=[],
    train_data_augmentations=[
        data_augmentation.scaling_pil(
            scale_range=(0.5, 1.5),
            copies=2
        ),
        data_augmentation.rotating(
            rotation_range=(-60, 60),
            padding_value=0,
            copies=2
        ),
    ],
    data_augmentations=[
        data_augmentation.flatten(),
        data_augmentation.normalize(),
        data_augmentation.one_hot_labels(num_classes=10)
    ],
    shuffle=True,
    batch_size=batch_size
)

model = SequentialModel(
    layers=[
        InputLayer(784),
        LinearLayer(392,
                    activation=relu(),
                    prev_weights_initializer=he_initializer()),
        LinearLayer(10,
                    activation=linear(),
                    prev_weights_initializer=he_initializer())
    ]
)
print("-----------------------------")
print("NN architecture + hyperparams")
print("-----------------------------\n")

model.summary()
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Max train epochs: {epochs}")

print("\n-----------------------------")
print("Train & Test process")
print("-----------------------------\n")

model.build(
    loss_function=mse(),
    optimizer=Adam(learning_rate=learning_rate),
    metrics=[
        AccuracyMetric(
            matching_function=one_hot_matching_function()
        ),
        LossMetric(
            loss_function=mse(),
            published_name='mse_loss'
        ),
    ],
)

model.fit(
    model_data_source=data_source,
    epochs=epochs,
    callbacks=[
        ProgressBarCallback(
            count_mode='batch',
            monitors=[
                'accuracy',
                'mse_loss',
            ]),
        ModelSaveCallback(
            monitor='accuracy',
            mode='max',
            monitor_save_threshold=0.98,
            filepath='mnist_greater_0.98accuracy',
        )
    ]
)

print("\n-----------------------------")
print("Load from file")
print("-----------------------------\n")

model = Model.load_from_file("mnist_greater_0.98accuracy")

model.test_epoch(
    test_data=data_source.test_data_batches(1),
    callbacks=[
        ProgressBarCallback(),
    ]
)
