import jax
import jax.numpy as jnp

import optax

import orbax.checkpoint as ocp

import tqdm
import time

from load_data import *

from transformer import get_model

import matplotlib.pyplot as plt

import numpy as np

# PARAMETER DEFINITIONS
EPOCHS = 10
main_rng = jax.random.PRNGKey(np.random.randint(0, 255))

loader63tr, loader63te, loader96tr, loader96te = get_dataloaders(shuffle=True)

model = get_model()
model96 = get_model(model_str="lorenz96")

example_src, example_target = next(iter(loader63tr))
example_src96, example_target96 = next(iter(loader96tr))

init_rngs = {
    "params": jax.random.PRNGKey(np.random.randint(0, 255)),
    "dropout": jax.random.PRNGKey(np.random.randint(0, 255)),
}
init_rngs2 = {
    'params': jax.random.PRNGKey(np.random.randint(0, 255)),
    'dropout': jax.random.PRNGKey(np.random.randint(0, 255)),
}

checkpoint_path = "/home/noah/Documents/Uni/DSML/final2/checkpoints/"

checkpointer = ocp.StandardCheckpointer()

modelvars = model.init(init_rngs, example_src, example_target)
model96vars = model96.init(init_rngs2, example_src96, example_target96)

@jax.jit
def loss_norm_fn(vars, src, tgt, rngs):
    preds = model96.apply(vars, src, tgt, rngs=rngs)
    return jnp.linalg.norm(jnp.linalg.norm(preds - tgt, axis=-1), axis=-1).mean()


@jax.jit
def loss_sum_fn(vars, src, tgt, rngs):
    preds = model96.apply(vars, src, tgt, rngs=rngs)
    return jnp.sum(jnp.linalg.norm(preds - tgt, axis=-1), axis=-1).mean()


lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-8,
    peak_value=1e-2,
    warmup_steps=5000,
    decay_steps=6250,
    end_value=1e-18,
    exponent=1.0,
)

optimizer = optax.adam(lr_schedule)
optimizer96 = optax.adam(lr_schedule)

loss_grad_norm = jax.jit(jax.value_and_grad(loss_norm_fn))
loss_grad_sum = jax.jit(jax.value_and_grad(loss_sum_fn))

opt_state = optimizer.init(modelvars)
opt_state2= optimizer96.init(model96vars)

loss_hist = []
loss_hist2 = []

for epoch in range(EPOCHS):
    loop = tqdm.tqdm(enumerate(loader96tr), total=len(loader96tr), leave=True, ncols=150)
    for i, (src, tgt) in loop:
        main_rng, dropout_rng = jax.random.split(main_rng)
        loss, grads = loss_grad_norm(model96vars, src, tgt, {'dropout': dropout_rng})
        loss_hist2.append(loss)

        loop.set_description_str(f"Epoch {epoch+1}: ")
        loop.set_postfix_str(f"current loss= {loss:.2f}")
    
        updates, opt_state2 = optimizer96.update(grads, opt_state2)
        model96vars = optax.apply_updates(model96vars, updates)

    checkpointer.save(checkpoint_path + str(time.time()) + "__96test1__" + str(epoch), model96vars)

