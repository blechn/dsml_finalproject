import numpy as np

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from psd import *

from load_data import *

l63tr, l63te, l96tr, l96te = get_datasets()

l63data = l63te.data[:10000]
l96data = l96te.data[:10000]
l63data = jnp.expand_dims(l63data, 0)
l96data = jnp.expand_dims(l96data, 0)
#print(l63data.shape, l96data.shape)

generated_sequence63 = jnp.load("./generated1.npy")
generated_sequence96 = jnp.load("./generated2.npy")
generated_sequence63 = jnp.expand_dims(generated_sequence63, 0)
generated_sequence96 = jnp.expand_dims(generated_sequence96, 0)
#print(generated_sequence63.shape, generated_sequence96.shape)

err63 = power_spectrum_error(generated_sequence63, l63data)
err96 = power_spectrum_error(generated_sequence96, l96data)

print(err63, err96)

spectrum_gen63 = get_average_spectrum(generated_sequence63)
spectrum_te63 = get_average_spectrum(l63data)

spectrum_gen96 = get_average_spectrum(generated_sequence96)
spectrum_te96 = get_average_spectrum(l96data)

print(spectrum_te63.shape)