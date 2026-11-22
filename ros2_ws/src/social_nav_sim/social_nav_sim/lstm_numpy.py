"""Run the MATLAB-trained LSTM predictor in plain numpy (copy of learning/predictor_numpy.py).

Loads weights_*.mat written by learning/matlab/train_predictor.m. Input is one
snippet in the person's own frame, time x features (see make_dataset.py):
  ego [8 x 4], nbr [8 x 4 x 5], rob [8 x 5]  ->  future [12 x 2] in metres.

`python learning/predictor_numpy.py <weights.mat> <snippets.mat>` checks the
numpy port against the test split (should match the MATLAB metrics).
"""
import sys

import numpy as np
from scipy.io import loadmat


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LSTMPredictor:
    def __init__(self, weights_mat):
        w = loadmat(weights_mat)
        self.mean = w['in_mean'].ravel()
        self.std = w['in_std'].ravel()
        self.W, self.R, self.b = w['lstm_W'], w['lstm_R'], w['lstm_b'].ravel()
        self.fc1_W, self.fc1_b = w['fc1_W'], w['fc1_b'].ravel()
        self.fc2_W, self.fc2_b = w['fc2_W'], w['fc2_b'].ravel()
        self.H = self.R.shape[1]
        self.n_features = self.W.shape[1]

    @staticmethod
    def features(ego, nbr=None, rob=None):
        """Stack inputs the way MATLAB's reshape did (column-major over people, then fields)."""
        parts = [ego]
        if nbr is not None:
            parts.append(np.transpose(nbr, (0, 2, 1)).reshape(nbr.shape[0], -1)
                         if nbr.ndim == 3 else nbr)
        if rob is not None:
            parts.append(rob)
        return np.concatenate(parts, axis=-1)

    def predict(self, x):
        """x: [T x F] or [N x T x F] -> [N x 12 x 2]."""
        if x.ndim == 2:
            x = x[None]
        x = (x - self.mean) / np.where(self.std > 0, self.std, 1.0)
        n, H = x.shape[0], self.H
        h, c = np.zeros((n, H)), np.zeros((n, H))
        for t in range(x.shape[1]):
            z = x[:, t] @ self.W.T + h @ self.R.T + self.b
            i, f, g, o = (sigmoid(z[:, :H]), sigmoid(z[:, H:2 * H]),
                          np.tanh(z[:, 2 * H:3 * H]), sigmoid(z[:, 3 * H:]))
            c = f * c + i * g
            h = o * np.tanh(c)
        y = np.maximum(h @ self.fc1_W.T + self.fc1_b, 0) @ self.fc2_W.T + self.fc2_b
        return np.stack([y[:, :12], y[:, 12:]], axis=-1)


def main():
    net = LSTMPredictor(sys.argv[1])
    d = loadmat(sys.argv[2])
    nbr = d['test_nbr']
    x = np.concatenate([d['test_ego'], np.transpose(nbr, (0, 1, 3, 2)).reshape(len(nbr), 8, -1)], -1)
    if net.n_features == x.shape[-1] + 5:
        x = np.concatenate([x, d['test_rob']], -1)
    elif net.n_features == 4:
        x = d['test_ego']
    pred = net.predict(x.astype(np.float64))
    err = np.linalg.norm(pred - d['test_fut'], axis=-1)
    near = d['test_near'].ravel() > 0
    print(f'numpy port: ADE {err.mean():.4f} m, FDE {err[:, -1].mean():.4f} m, '
          f'ADE near robot {err[near].mean():.4f} m')


if __name__ == '__main__':
    main()
