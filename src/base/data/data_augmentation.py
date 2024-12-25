from PIL import Image

import random
import numpy as np
from scipy import ndimage


class DataAugmentation:
    def __init__(
            self,
            augment_function,
            name: str = "DataAugmentation",
    ):
        self.__augment_function = augment_function
        self.name = name
        pass

    def __call__(self, data_tuple: (np.array, np.array)) -> [(np.array, np.array)]:
        answer = self.__augment_function(data_tuple)
        if not isinstance(answer, list):
            return [answer]
        return answer


def __add_padding(current_img: np.array, required_shape: (int, int), padding_value: float = 0.) -> np.array:
    current_height, current_width = current_img.shape
    required_height, required_width = required_shape

    if current_height == required_height and current_width == required_width:
        return current_img

    height_to_be_filled, width_to_be_filled = (required_shape[0] - current_img.shape[0],
                                               required_shape[1] - current_img.shape[1])

    image_padding = (
        (height_to_be_filled // 2, height_to_be_filled - (height_to_be_filled // 2)),
        (width_to_be_filled // 2, width_to_be_filled - (width_to_be_filled // 2)),
    )

    return np.pad(current_img, image_padding, mode='constant', constant_values=padding_value)


def flatten():
    def augment_function(data_tuple):
        return np.array(data_tuple[0]).flatten(), data_tuple[1]

    return DataAugmentation(augment_function, name="flatten")


def normalize():
    def augment_function(data_tuple):
        x = np.array(data_tuple[0])
        if np.max(x) == 0:
            return [], []
        return x / np.max(x), data_tuple[1]

    return DataAugmentation(augment_function, name="normalize")


def one_hot_labels(num_classes=10):
    def augment_function(data_tuple):
        y = np.zeros(num_classes)
        y[data_tuple[1] - 1] = 1
        return data_tuple[0], y

    return DataAugmentation(augment_function, name="one_hot_labels")


def scaling_pil(
        scale_range: (float, float) = (0.5, 1.5),
        copies=0,
        padding_value: float = 0.):
    def augment_function(data_tuple):
        img, label = np.array(data_tuple[0]), data_tuple[1]
        img_height, img_width = img.shape

        img_normalization = np.max(img)
        pil_normalized_image = np.array(img, dtype=np.uint8)

        min_scale, max_scale = scale_range

        scale_values = [random.uniform(min_scale, max_scale) for _ in range(copies)] or [
            random.uniform(min_scale, max_scale)]

        augmented_datas = [(img, label)]
        for scale_value in scale_values:
            new_img_height, new_img_width = int(img_height * scale_value), int(img_width * scale_value)
            new_img = (Image.fromarray(pil_normalized_image).resize((new_img_height, new_img_width)))

            if scale_value > 1:
                top_left_position = ((new_img_height - img_height) // 2, (new_img_width - img_width) // 2)
                bottom_right_position = (top_left_position[0] + img_height, top_left_position[1] + img_width)
                new_img = new_img.crop((top_left_position[1], top_left_position[0],
                                        bottom_right_position[1], bottom_right_position[0]))
            new_img = np.array(new_img) / (255 / img_normalization)
            new_img = __add_padding(new_img, required_shape=img.shape, padding_value=padding_value)

            if len(scale_values) > 1:
                augmented_datas.append((new_img, label))
            else:
                augmented_datas[0] = (new_img, label)

        return augmented_datas

    return DataAugmentation(augment_function, name="scaling_pil")


def rotating(
        rotation_range: (float, float) = (-30, 30),
        copies=0,
        padding_value: float = 0.):
    def augment_function(data_tuple):
        img, label = np.array(data_tuple[0]), data_tuple[1]
        min_rotation, max_rotation = rotation_range

        rotation_values = [random.uniform(min_rotation, max_rotation) for _ in range(copies)] or [
            random.uniform(min_rotation, max_rotation)]

        augmented_datas = [(img, label)]

        for rotation_value in rotation_values:
            rotated_img = ndimage.rotate(img, angle=rotation_value, reshape=False, cval=padding_value)
            if len(rotation_values) > 1:
                augmented_datas.append((rotated_img, label))
            else:
                augmented_datas[0] = (rotated_img, label)

        return augmented_datas

    return DataAugmentation(augment_function, name="rotating")
