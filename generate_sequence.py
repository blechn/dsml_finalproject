import jax
import jax.numpy as jnp

import flax

import orbax.checkpoint as ocp

from transformer import *

import tqdm

from load_data import *

import matplotlib.pyplot as plt

l63tr, l63te, l96tr, l96te = get_datasets()

ckpt63 = ocp.StandardCheckpointer()
ckpt96 = ocp.StandardCheckpointer()

model63 = get_model()
model96 = get_model("lorenz96")

l63_10_ckpt_path= "/home/noah/Documents/Uni/DSML/final2/checkpoints/1709823961.9568238__63test1__9"

l96_10_ckpt_path = "/home/noah/Documents/Uni/DSML/final2/checkpoints/1709826158.579993__96test1__9"

model63vars_10 = ckpt63.restore(l63_10_ckpt_path)

model96vars_10 = ckpt96.restore(l96_10_ckpt_path)

def predict_sequence(model, vars, input_sequence, prediction_length=100000):
    seq = jnp.zeros(shape=(prediction_length, model.d_model))

    seq = seq.at[0:input_sequence.shape[0]].set(input_sequence)

    for i in tqdm.trange(prediction_length-101):
        enc_input = seq[i:i+100]
        dec_input = seq[i+1:i+101]
        rngs = {'dropout': jax.random.PRNGKey(0)}  
        prediction = model.apply(vars, enc_input, dec_input, rngs=rngs, train=False)
        seq = seq.at[i+101].set(prediction[0][-1])

    return seq

in1 = l63te[0][0]
in2 = l96te[0][0]
test2 = predict_sequence(model63, model63vars_10, in1, prediction_length=10000)
test = predict_sequence(model96, model96vars_10, in2, prediction_length=10000)

x = test2[:, 0]
y = test2[:, 1]
z = test2[:, 2]

jnp.save("./generated1", test2)
jnp.save("./generated2", test)