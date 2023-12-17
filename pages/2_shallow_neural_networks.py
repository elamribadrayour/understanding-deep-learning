import jax.nn
import graphviz
import jax.numpy
import jax.random
from jax import Array
from torchvision.datasets import MNIST

import streamlit
from stqdm import stqdm
from matplotlib import pyplot as plt


def get_nd_array(x) -> Array:
    return jax.numpy.asarray(x.numpy())


def get_flattened_data(x: Array) -> Array:
    return jax.numpy.reshape(x, (x.shape[0], int(x.shape[1] * x.shape[2])))


def get_batched_data(x: Array, batch_size: int = 32) -> Array:
    batches = jax.numpy.array_split(x, len(x) // batch_size)
    batches = [batch for batch in batches if len(batch) == batch_size]
    return jax.numpy.stack(batches, axis=0)


def get_data(batch_size: int) -> tuple[Array, Array, Array, Array]:
    data = MNIST(root="./.cache", train=True, download=True)

    x_test = get_nd_array(data.test_data) / 255.0
    x_train = get_nd_array(data.train_data) / 255.0

    y_train = get_nd_array(data.train_labels)
    y_test = get_nd_array(data.test_labels)

    y_test = jax.nn.one_hot(x=y_test, num_classes=10)
    y_train = jax.nn.one_hot(x=y_train, num_classes=10)

    x_test = get_flattened_data(x=x_test)
    x_train = get_flattened_data(x=x_train)

    x_test_batch = get_batched_data(x=x_test, batch_size=batch_size)
    y_test_batch = get_batched_data(x=y_test, batch_size=batch_size)

    x_train_batch = get_batched_data(x=x_train, batch_size=batch_size)
    y_train_batch = get_batched_data(x=y_train, batch_size=batch_size)

    return x_train_batch, x_test_batch, y_train_batch, y_test_batch


def get_layer(m: int, n: int, k: jax.Array, scale: float) -> tuple[Array, Array]:
    w_key, b_key = jax.random.split(k)
    bias = scale * jax.random.normal(b_key, (n,))
    weights = scale * jax.random.normal(w_key, (n, m))
    return weights, bias


def get_model() -> list[tuple[Array, Array]]:
    scale = 1e-2
    sizes = [28 * 28, 256, 10]
    key = jax.random.PRNGKey(seed=0)
    keys = jax.random.split(key, len(sizes))

    layers = list()
    for m, n, k in zip(sizes[:-1], sizes[1:], keys):
        layers.append(get_layer(m=m, n=n, k=k, scale=scale))
    return layers


def forward(model: list[tuple[Array, Array]], x: Array):
    """Function for per-example predictions."""
    outputs = x
    for w, b in model[:-1]:
        outputs = jax.numpy.dot(w, outputs) + b
        outputs = jax.nn.swish(outputs)
    final_w, final_b = model[-1]
    logits = jax.numpy.dot(final_w, outputs) + final_b
    return logits


batch_forward = jax.vmap(forward, in_axes=(None, 0))


def loss(model: list[tuple[Array, Array]], x: Array, y: Array) -> Array:
    """Categorical cross entropy."""
    logits = batch_forward(model, x)
    # logits = jax.numpy.exp(logits)
    # denominator = jax.numpy.sum(logits, axis=1)
    # logits = jax.numpy.vstack(
    #     [logits[i, :] / denominator[i] for i in range(len(logits))]
    # )
    log_preds = logits - jax.nn.logsumexp(logits)

    return -jax.numpy.mean(y * log_preds)


@jax.jit
def update(
    x: Array,
    y: Array,
    epoch: int,
    model: list[tuple[Array, Array]],
) -> tuple[list[tuple[Array, Array]], float]:
    decay = 5
    decay_rate = 0.95
    lr = decay_rate ** (epoch / decay)
    loss_, grads = jax.value_and_grad(loss)(model, x, y)
    model_out = list()
    for (w, b), (dw, db) in zip(model, grads):
        model_out.append((w - lr * dw, b - lr * db))
    return model_out, loss_


@jax.jit
def get_batch_accuracy(model, x, y):
    y = jax.numpy.argmax(y, axis=1)
    predicted_class = jax.numpy.argmax(batch_forward(model, x), axis=1)
    return jax.numpy.mean(predicted_class == y)


def get_accuracy(model: list[tuple[Array, Array]], x: Array, y: Array):
    accs = []
    for images, targets in zip(x, y):
        accs.append(get_batch_accuracy(model, images, targets))
    return jax.numpy.mean(jax.numpy.array(accs))


def train(
    x: Array,
    y: Array,
    nb_epochs: int,
    batch_size: int,
    model: list[tuple[Array, Array]],
) -> dict:
    losses = list()
    accuracies = list()
    for epoch in stqdm(range(nb_epochs)):
        for i in range(batch_size):
            model, loss_ = update(
                x=x[i],
                y=y[i],
                model=model,
                epoch=epoch,
            )
        losses.append(loss_)
        accuracy = get_accuracy(model=model, x=x, y=y)
        accuracies.append(accuracy)
    return {
        "model": model,
        "loss": losses,
        "accuracy": accuracies,
    }


def predict(model: list[tuple[Array, Array]], image: Array) -> Array:
    output = forward(model=model, x=image)
    return output.argmax(axis=0)


def plot_data(x: Array, y: Array) -> None:
    streamlit.markdown("## Dataset")
    fig, axes = plt.subplots(nrows=2, ncols=4)
    fig.set_facecolor("#faf9f9")
    k = 0
    for i in range(2):
        for j in range(4):
            target = jax.numpy.argmax(y[5][k])
            image = jax.numpy.reshape(x[5][k, ...], (28, 28))
            axes[i, j].imshow(X=image)
            axes[i, j].set_title(target)
            axes[i, j].axis("off")
            axes[i, j].set_facecolor("#faf9f9")
            k = k + 1
    streamlit.pyplot(fig=fig)


def plot_model() -> None:
    streamlit.markdown("## Model")
    graph = graphviz.Graph()
    graph.edge("image | 28*28", "input | 784")
    graph.edge("input | 784", "hidden layer | 256")
    graph.edge("hidden layer | 256", "output | 10")
    streamlit.columns(3)[1].graphviz_chart(graph)


def plot_artifacts(accuracies: list[float], losses: list[float]) -> None:
    streamlit.markdown("## Training Artifacts")

    loss_ = round(float(losses[-1]), 2)
    acc = int(100 * float(accuracies[-1]))
    columns = streamlit.columns(2)
    columns[0].metric(label="accuracy", value=f"{acc}%")
    columns[1].metric(label="loss", value=f"{loss_}")

    fig, ax = plt.subplots()
    fig.set_facecolor("#faf9f9")
    ax.plot(accuracies)
    ax.set_title("Accuracies")
    streamlit.pyplot(fig=fig)

    fig, ax = plt.subplots()
    fig.set_facecolor("#faf9f9")
    ax.plot(losses)
    ax.set_title("Losses")
    streamlit.pyplot(fig=fig)


streamlit.markdown("# Multi Layer Perceptron")

streamlit.sidebar.markdown("# Data")
batch_size = int(
    streamlit.sidebar.number_input(
        step=16,
        value=32,
        min_value=1,
        max_value=128,
        label="batch size",
    )
)

streamlit.sidebar.markdown("# Model")
nb_epochs = int(
    streamlit.sidebar.number_input(
        step=10,
        value=20,
        min_value=10,
        max_value=500,
        label="nb epochs",
    )
)

is_train = streamlit.sidebar.columns(3)[1].button(
    label="train",
)

x_train, x_test, y_train, y_test = get_data(batch_size=batch_size)

plot_data(x=x_train, y=y_train)

model = get_model()

plot_model()

if is_train is False:
    streamlit.stop()

output = train(
    x=x_train,
    y=y_train,
    model=model,
    nb_epochs=nb_epochs,
    batch_size=batch_size,
)

plot_artifacts(accuracies=output["accuracy"], losses=output["loss"])
