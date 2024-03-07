This is the repo of my final project of the Lecture "Dynamical Systems Theory in Machine Learning".  

It is written completely in the JAX/FLAX ecosystem as I have no access to a GPU and JAX proved to be much faster on my CPU than PyTorch.

If you want to execute the code, you need to modify a bit.
1. In `train.py` you need to change the specific model and model variables as well as the optimizer in the training loop at the end of the file for the specific system. You only need to change the names in the training loop, everything else is already initialized correctly!
2. In `generate_sequences.py` you need to modify the paths of the checkpoints since I used the absolute paths on my computer, and they will most likely be different on other machines (unless you have the exact same folder structure as me).

If you want to change the hyperparameters of the transformer model, you can just initialize a new object of the Transformer class instead of using the `get_model` function, since it uses my "default" settings which still worked on my laptop.
