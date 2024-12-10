from v1.src.base.activation import *
from v1.src.base.callbacks.default_callbacks import ProgressBarCallback
from v1.src.base.data import data_augmentation, ModelDataSource
from v1.src.base.layers import InputLayer
from v1.src.base.layers.linear_layer import *
from v1.src.base.loss_function import mse
from v1.src.base.metrics import AccuracyMetric, one_hot_matching_function
from v1.src.base.models import SequentialModel
from v1.src.base.optimizers.adam import Adam
from v1.src.base.value_initializer import xavier_initializer
from v1.src.mnist.mnist_dataloader import MnistDataloader
from v1.src.base.models.model import Model
import matplotlib.pyplot as plt

mnist_dataloader = MnistDataloader(
    training_images_filepath = "./v1/src/mnist/data/train-images-idx3-ubyte/train-images-idx3-ubyte",
    training_labels_filepath = "./v1/src/mnist/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
    test_images_filepath = "./v1/src/mnist/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    test_labels_filepath = "./v1/src/mnist/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

batch_size = 200

data_source = ModelDataSource(
    test_data=(x_test[:100], y_test[:100]),
    data_augmentations=[
        data_augmentation.scaling_pil(
            copies=1,
            scale_range=(0.5, 1.5)
        ),
        data_augmentation.rotating(
            rotation_range=(-60, 60),
            copies=1,
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
    data_augmentations=[
        data_augmentation.scaling_pil(
            copies=1,
            scale_range=(0.5, 1.5)
        ),
        data_augmentation.rotating(
            rotation_range=(-60, 60),
            copies=1,
            padding_value=0),
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
        LinearLayer(300,
                    activation=relu(),
                    prev_weights_initializer=xavier_initializer()),
        LinearLayer(10,
                    activation=linear(),
                    prev_weights_initializer=xavier_initializer())
    ]
)
print("-----------------------------")
print("NN architecture + hyperparams")
print("-----------------------------\n")

model.summary()
print(f"Learning rate: {0.0013}")
print(f"Batch size: {batch_size}")
print(f"Max train epochs: {20}")

print("\n-----------------------------")
print("Train & Test process")
print("-----------------------------\n")

model.build(
    loss_function=mse(),
    optimizer=Adam(learning_rate=0.0013),
    metrics=[
        AccuracyMetric(matching_function=one_hot_matching_function())
    ]
)

model.fit(
    model_data_source=data_source,
    epochs=20,
    callbacks=[
        ProgressBarCallback(),
    ],
)

print("\n-----------------------------")
print("Save & Load from file")
print("-----------------------------\n")

model.save_to_file("2_hl_Adam_lr0.0013.txt")
model = Model.load_from_file("2_hl_Adam_lr0.0013.txt")

model.test_epoch(
    test_data=data_source.test_data_batches(1),
    callbacks=[
        ProgressBarCallback(),
    ]
)