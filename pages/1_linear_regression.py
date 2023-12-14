import jax
from jax.numpy import ndarray

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

import streamlit
from stqdm import stqdm
from matplotlib import pyplot as plt


streamlit.title("1 Feature Linear Regression")


@streamlit.cache_data()
def get_datasets(nb_features: int) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    x, y = fetch_california_housing(as_frame=True, return_X_y=True)

    features = x.to_numpy()[:, :nb_features]

    target = y.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.4)
    x_train = jax.numpy.asarray(x_train)
    x_test = jax.numpy.asarray(x_test)
    y_train = jax.numpy.asarray(y_train)
    y_test = jax.numpy.asarray(y_test)
    return x_train, x_test, y_train, y_test


def get_scalers(x: ndarray) -> tuple[ndarray, ndarray]:
    means = jax.numpy.mean(x, axis=0)
    variances = jax.numpy.var(x, axis=0)
    return means, variances


def get_model(size: int) -> tuple[ndarray, ndarray]:
    bias = jax.numpy.zeros(shape=(1,))
    weights = jax.numpy.zeros(shape=(size,))
    return weights, bias


def forward(x: ndarray, weights: ndarray, bias: ndarray) -> ndarray:
    return jax.numpy.dot(x, weights) + bias


def loss_fn(weights: ndarray, bias: ndarray, x: ndarray, y: ndarray) -> ndarray:
    y_prediction = forward(x=x, weights=weights, bias=bias)
    errors = y_prediction - y
    return jax.numpy.mean(errors**2)


def update(
    weights: ndarray, bias: ndarray, dw: ndarray, db: ndarray, learning_rate: float
):
    return weights - learning_rate * dw, bias - learning_rate * db


def train(
    n_epochs: int,
    bias: ndarray,
    x_train: ndarray,
    y_train: ndarray,
    weights: ndarray,
    learning_rate: float,
):
    grad_w = jax.jit(jax.grad(fun=loss_fn, argnums=0))
    grad_b = jax.jit(jax.grad(fun=loss_fn, argnums=1))

    y_pred = forward(weights=weights, bias=bias, x=x_train)
    loss = float(loss_fn(weights=weights, bias=bias, x=x_train, y=y_train))

    y_preds = [y_pred]
    losses = [round(loss, 2)]

    for _ in stqdm(range(n_epochs)):
        dw = grad_w(weights, bias, x_train, y_train)
        db = grad_b(weights, bias, x_train, y_train)
        weights, bias = update(
            weights=weights, bias=bias, dw=dw, db=db, learning_rate=learning_rate
        )
        loss = float(loss_fn(x=x_train, y=y_train, weights=weights, bias=bias))
        y_pred = forward(x=x_train, weights=weights, bias=bias)
        losses.append(round(loss, 2))
        y_preds.append(y_pred)

    return weights, bias, {"prediction": y_preds, "loss": losses}


def test(x: ndarray, y: ndarray, weights: ndarray, bias: ndarray) -> float:
    loss = loss_fn(weights=weights, bias=bias, x=x, y=y)
    return round(float(loss), 2)


n_epochs = int(streamlit.sidebar.number_input(label="epochs", value=1000))
learning_rate = streamlit.sidebar.number_input(label="learning rate", value=1e-2)
nb_features = int(
    streamlit.sidebar.number_input(
        label="nb features", value=1, step=1, min_value=1, max_value=8
    )
)
correlation_idx = int(
    streamlit.sidebar.number_input(
        label="correlation idx", value=0, step=1, max_value=nb_features - 1
    )
)

x_train, x_test, y_train, y_test = get_datasets(nb_features=nb_features)

means, variances = get_scalers(x=x_train)
x_test = (x_test - means) / variances
x_train = (x_train - means) / variances

size = x_train.shape[1]
weights, bias = get_model(size=size)
weights, bias, artifacts = train(
    bias=bias,
    weights=weights,
    x_train=x_train,
    y_train=y_train,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
)

test_loss = test(x=x_test, y=y_test, weights=weights, bias=bias)

columns = streamlit.columns(2)
delta = round(test_loss - artifacts["loss"][-1], 2)
columns[0].metric(label="train loss", value=artifacts["loss"][-1])
columns[1].metric(
    label="test loss", value=test_loss, delta=delta, delta_color="inverse"
)

streamlit.markdown("## Training loss")
fig, ax = plt.subplots()
fig.set_facecolor("#faf9f9")
ax.set_facecolor("#faf9f9")
plt.plot(range(n_epochs + 1), artifacts["loss"], "#89b0ae")
streamlit.pyplot(fig=fig)


streamlit.markdown("## Prediction vs Real")
fig, ax = plt.subplots()
fig.set_facecolor("#faf9f9")
ax.set_facecolor("#faf9f9")
plt.scatter(x=x_train[:, correlation_idx], y=y_train, marker=".", c="#ffd6ba")
plt.scatter(
    x=x_train[:, correlation_idx],
    y=artifacts["prediction"][-1],
    marker=".",
    c="#89b0ae",
)
streamlit.pyplot(fig=fig)
