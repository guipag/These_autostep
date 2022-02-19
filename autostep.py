import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def autostep(inp, out, nbIn, nFir):
    nbPts = min(inp.shape[1], len(out))

    inAcc = np.zeros((nbIn, nFir))
    outAcc = np.zeros(nFir)

    e = np.zeros(nbPts)
    kappa = 0.027655
    gamma = 0.22104

    mu = np.ones((nbIn, nFir)) / (nFir * nbIn + 1)
    h = np.zeros((nbIn, nFir))
    w = np.zeros((nbIn, nFir))
    v = np.zeros((nbIn, nFir))

    for n in tqdm(np.arange(nbPts)):
        inAcc = np.roll(inAcc, 1, axis=1)
        inAcc[:, 1] = inp[:, n]

        outAcc = np.roll(outAcc, -1)
        outAcc[-1] = out[n]

        e[n] = outAcc[-1] - sum(sum(h * inAcc))

        v = np.maximum(abs(e[n] * inAcc * w), v + gamma * mu * abs(inAcc)**2 * (abs(e[n] * inAcc * w) - v))

        slice_v0 = (v == 0)
        v[slice_v0] = 1
        alpha = np.log(mu) + kappa * e[n] * inAcc * w / v
        alpha[slice_v0] = np.log(mu[slice_v0])

        mu = np.exp(alpha)
        mu = mu / np.maximum(mu * inAcc**2, 1)

        h = h + mu * e[n] * inAcc

        w = (1 - mu * abs(inAcc)**2) * w + mu * e[n] * inAcc

    return h, e


if __name__ == "__main__":
    h1 = np.zeros(100)
    h2 = np.zeros(100)

    h1[20] = 0.9
    h2[30] = 0.8

    x = np.zeros((2, 100000))
    x[0,:] = np.random.normal(0, 1, size=100000)
    x[1,:] = np.random.normal(0, 1, size=100000)

    y = np.convolve(h1, x[0,:]) + np.convolve(h2, x[1,:])

    h, e = autostep(x, y, 2, 100)

    plt.figure()
    plt.plot(h[1,:])
    plt.show()

