import numpy as np
import torch
from skimage.transform import resize
from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01, one_hot_target

x = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
y = np.random.randint(1, 5, size=(256, 256), dtype=np.uint8)

transform = ComposeDouble([
    FunctionWrapperDouble(normalize_01),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(one_hot_target, input=False, target=True)
    ])

x_t, y_t = transform(x, y)

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'x_t: shape: {x_t.shape}  type: {x_t.dtype}')
print(f'x_t = min: {x_t.min()}; max: {x_t.max()}')

print(f'y = shape: {y.shape}; class: {np.unique(y)}')
print(f'y_t = shape: {y_t.shape}; class: {np.unique(y_t)}')