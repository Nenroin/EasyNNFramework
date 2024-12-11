import numpy as np

from src.base.data.data_augmentation import DataAugmentation
from src.base.data.data_batch_wrapper import DataBatchWrapper

class ModelDataSource:
    def __init__(
            self,
            train_data: (np.array, np.array) = None,
            train_data_augmentations: [DataAugmentation] = None,
            test_data: (np.array, np.array) = None,
            test_data_augmentations: [DataAugmentation] = None,
            data_augmentations: [DataAugmentation] = None,
            shuffle=False,
            batch_size=1,
    ):
        self.data_augmentations = data_augmentations or []

        self.test_data_augmentations = (test_data_augmentations or []) + self.data_augmentations

        self.__test_data = test_data or ([], [])
        self.__test_data = self.proceed_augmentations(self.__test_data, self.test_data_augmentations)

        self.train_data_augmentations = (train_data_augmentations or []) + self.data_augmentations

        self.__train_data = train_data or ([], [])
        self.__train_data = self.proceed_augmentations(self.__train_data, self.train_data_augmentations)

        self.batch_size = batch_size

        shuffle and self.__make_shuffle()

    def train_data_batches(self, batch_size = None):
        batch_size = batch_size or self.batch_size
        return DataBatchWrapper(self.__train_data, batch_size)

    def test_data_batches(self, batch_size = None):
        batch_size = batch_size or self.batch_size
        return DataBatchWrapper(self.__test_data, batch_size)

    def proceed_augmentations(self, data, data_augmentations):
        for augmentation in data_augmentations:
            data = self.proceed_augmentation(data, augmentation)
        return data

    @classmethod
    def proceed_augmentation(cls, data, data_augmentation):
        (data_x, data_y) = data
        new_data_x = []
        new_data_y = []
        for i, _ in enumerate(data_x):
            augmented_datas = data_augmentation((data_x[i], data_y[i]))
            for augmented_data in augmented_datas:
                augmented_data_x, augmented_data_y = augmented_data
                new_data_x.append(augmented_data_x)
                new_data_y.append(augmented_data_y)
        return new_data_x, new_data_y

    def __make_shuffle(self):
        self.__test_data = self.__shuffle_datas_tuple(self.__test_data)
        self.__train_data = self.__shuffle_datas_tuple(self.__train_data)

    @classmethod
    def __shuffle_datas_tuple(cls, datas_tuple):
        rng = np.random.default_rng()
        list_of_tuples = list(zip(datas_tuple[0], datas_tuple[1]))
        if len(list_of_tuples) > 0:
            rng.shuffle(list_of_tuples)
            x, y = zip(*list_of_tuples)
            return x, y
        else:
            return datas_tuple