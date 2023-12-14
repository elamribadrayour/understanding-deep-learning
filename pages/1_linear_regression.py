import jax
from jax.numpy import ndarray

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

import seaborn
import streamlit
from stqdm import stqdm
from matplotlib import pyplot as plt


streamlit.title("1 Feature Linear Regression")


@streamlit.cache_data()
def get_datasets() -> tuple[ndarray, ndarray, ndarray, ndarray]:
    x, y = fetch_california_housing(as_frame=True, return_X_y=True)
    features = x.to_numpy()[:, :1]
    target = y.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.4)
    x_train = jax.numpy.asarray(x_train)
    x_test = jax.numpy.asarray(x_test)
    y_train = jax.numpy.asarray(y_train)
    y_test = jax.numpy.asarray(y_test)
    return x_train, x_test, y_train, y_test


def get_model(size: int) -> tuple[ndarray, ndarray]:
    bias = jax.numpy.zeros(shape=(1,))
    weights = jax.numpy.zeros(shape=(size,))
    return weights, bias


def forward(x: ndarray, weights: ndarray, bias: ndarray) -> ndarray:
    return jax.numpy.dot(x, weights) + bias


def loss_fn(
    weights: ndarray, bias: ndarray, x_train: ndarray, y_train: ndarray
) -> ndarray:
    y_prediction = forward(x=x_train, weights=weights, bias=bias)
    errors = y_prediction - y_train
    return jax.numpy.mean(errors**2)


def update(
    weights: ndarray, bias: ndarray, dw: ndarray, db: ndarray, learning_rate: float
):
    return weights - learning_rate * dw, bias - learning_rate * db


@streamlit.cache_data()
def train(
    n_epochs: int,
    _bias: ndarray,
    _x_train: ndarray,
    _y_train: ndarray,
    _weights: ndarray,
    learning_rate: float,
):
    grad_w = jax.grad(fun=loss_fn, argnums=0)
    grad_b = jax.grad(fun=loss_fn, argnums=1)

    y_pred = forward(weights=_weights, bias=_bias, x=_x_train)
    loss = float(
        loss_fn(weights=_weights, bias=_bias, x_train=_x_train, y_train=_y_train)
    )

    y_preds = [y_pred]
    losses = [round(loss, 2)]

    for _ in stqdm(range(n_epochs)):
        dw = grad_w(_weights, _bias, _x_train, _y_train)
        db = grad_b(_weights, _bias, _x_train, _y_train)
        _weights, _bias = update(
            weights=_weights, bias=_bias, dw=dw, db=db, learning_rate=learning_rate
        )
        loss = float(
            loss_fn(x_train=_x_train, y_train=_y_train, weights=_weights, bias=_bias)
        )
        y_pred = forward(x=_x_train, weights=_weights, bias=_bias)
        losses.append(round(loss, 2))
        y_preds.append(y_pred)

    return _weights, _bias, {"prediction": y_preds, "loss": losses}


n_epochs = int(streamlit.sidebar.number_input(label="epochs", value=5))
learning_rate = streamlit.sidebar.number_input(label="learning rate", value=1e-3)

x_train, x_test, y_train, y_test = get_datasets()

size = x_train.shape[1]
weights, bias = get_model(size=size)
weights, bias, artifacts = train(
    _bias=bias,
    _weights=weights,
    _x_train=x_train,
    _y_train=y_train,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
)

fig, ax = plt.subplots()
seaborn.scatterplot(x=x_train[:, 0], y=y_train, ax=ax)
# for y_pred in stqdm(artifacts["prediction"]):
#     seaborn.lineplot(x=x_train[:, 0], y=y_pred)
streamlit.pyplot(fig=fig)

fig, ax = plt.subplots()
y = artifacts["loss"]
x = range(len(y))
seaborn.scatterplot(x=x, y=y, ax=ax)
streamlit.pyplot(fig=fig)
