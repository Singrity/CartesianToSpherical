import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy


def image_load(path):
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype='int32')
    return data


def convert_to_spherical(matrix):
    matrix = matrix / 255
    x_data = matrix[:, :, 0]
    y_data = matrix[:, :, 1]
    z_data = matrix[:, :, 2]

    r = np.sqrt(np.power(x_data, 2) + np.power(y_data, 2) + np.power(z_data, 2))
    theta = np.arccos(z_data / r)
    alpha = np.arctan2(y_data, x_data)

    print(matrix.dtype)

    matrix[:, :, 0] = r
    matrix[:, :, 1] = theta
    matrix[:, :, 2] = alpha

    print(matrix.dtype)
    return matrix


def convert_to_certain(matrix):
    r = matrix[:, :, 0]
    theta = matrix[:, :, 1]
    alpha = matrix[:, :, 2]

    x_data = r * np.cos(alpha) * np.sin(theta)
    y_data = r * np.sin(alpha) * np.sin(theta)
    z_data = r * np.cos(theta)

    matrix[:, :, 0] = x_data
    matrix[:, :, 1] = y_data
    matrix[:, :, 2] = z_data
    matrix = (matrix * 255).astype('int32')

    return matrix


def change_r(matrix, amount):
    matrix[:, :, 0] *= amount
    return matrix


def change_theta(matrix, amount):
    matrix[:, :, 1] *= amount
    return matrix


def change_alpha(matrix, amount):
    matrix[:, :, 2] *= amount
    return matrix


image_matrix = image_load('../data/example.png')

# original image

plt.imshow(image_matrix)
plt.show()


sph_img = convert_to_spherical(image_matrix)

# from spherical to certain
new_image = convert_to_certain(sph_img)
plt.imshow(new_image)
plt.show()


chanced_r_image = convert_to_certain(change_r(convert_to_spherical(image_matrix), 9))
plt.imshow(chanced_r_image)
plt.show()

chanced_theta_image = convert_to_certain(change_theta(convert_to_spherical(image_matrix), 3.14))
plt.imshow(chanced_theta_image)
plt.show()

chanced_alpha_image = convert_to_certain(change_alpha(convert_to_spherical(image_matrix), 90))
plt.imshow(chanced_alpha_image)
plt.show()

