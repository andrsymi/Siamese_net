
"""
Tests for the siamese neural network module
"""

import numpy as np
import pandas as pd
import keras
from keras import Model, Input
from keras.layers import Concatenate, Dense, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from siamese import SiameseNetwork


def create_base_model(input_shape):
    # Define base
    model_input = Input(shape=input_shape)

    embedding = Dense(4)(model_input)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)

    return Model(model_input, embedding)


def create_head_model(embedding_shape):
    # Define head model
    embedding_a = Input(shape=embedding_shape)
    embedding_b = Input(shape=embedding_shape)

    head = Concatenate()([embedding_a, embedding_b])
    head = Dense(4)(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    head = Dense(1)(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    return Model([embedding_a, embedding_b], head)


def test_siamese():
    """
    Test that all components the siamese network work correctly by executing a
    training run against generated data.
    """

    # Read data
    df = pd.read_csv("data.csv")
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    x = df.iloc[:, 1:]
    y = df.label

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

    num_classes = len(df.label.unique())
    input_shape = (x_train.shape[1],)
    epochs = 1000

    # Create siamese neural network
    base_model = create_base_model(input_shape)
    head_model = create_head_model(base_model.output_shape)
    siamese_network = SiameseNetwork(base_model, head_model)

    # Prepare siamese network for training
    siamese_network.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam())

    # Evaluate network before training to establish a baseline
    score_before = siamese_network.evaluate_generator(
        x_train, y_train, batch_size=64
    )

    # Train network
    siamese_network.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=64,
                        epochs=epochs)

    # Evaluate network
    score_after = siamese_network.evaluate(x_train, y_train, batch_size=64)

    # Ensure that the training loss score improved as a result of the training
    assert(score_before > score_after)


if __name__ == "__main__":
    test_siamese()
