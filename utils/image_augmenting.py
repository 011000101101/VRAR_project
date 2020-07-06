import numpy as np


def add_noise_greyscale(image: np.ndarray, noise_typ: str):
    """
    source: https://stackoverflow.com/a/30609854
    adds variable noise to an image
    :param noise_typ: One of the following strings, selecting the type of noise to add:
                        'gauss'     Gaussian-distributed additive noise.
                        'poisson'   Poisson-distributed noise generated from the data.
                        's_p'       Replaces random pixels with 0 or 1.
                        'speckle'   Multiplicative noise using out = image + n*image,where
                                    n is uniform noise with specified mean & variance.
    :param image: Input image data. Will be converted to float.
    :return:
    """
    if noise_typ == "gauss":
        shape = image.shape
        mean = 0.02
        var = 0.2
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, shape)
        gauss = gauss.reshape(shape)
        noisy = image + gauss
        return noisy.clip(0, 255)
    elif noise_typ == "s_p":
        s_vs_p = 0.5
        amount = 0.002  # 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out.clip(0, 255)
    elif noise_typ == "speckle":
        w, h = image.shape
        gauss = np.random.randn(w, h)
        gauss = gauss.reshape((w, h))
        gauss = gauss / 3
        noisy = image + image * gauss
        return noisy.clip(0, 255)