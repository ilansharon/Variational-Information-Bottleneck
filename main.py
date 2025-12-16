from models import MLP, MLPVIB

import torch
from torch import optim

from training import getLoaders, getDevice, setSeed, trainRun


def main():
    setSeed(0)

    device = getDevice()
    print("Using device:", device)

    batch = 128
    lr = 1e-3
    numEpochs = 50
    beta = 1e-3

    trainLoad, testLoad = getLoaders(batch=batch)

    modelBase = MLP()
    modelBase.to(device)
    z_size = modelBase.fcz.out_features
    optimizerBase = optim.Adam(modelBase.parameters(), lr=lr)

    historyBase = trainRun(
        modelBase,
        trainLoad,
        testLoad,
        optimizerBase,
        device,
        numEpochs,
        beta=None,
        tag="BASE",
        meta={
            "model": "MLP",
            "batch": batch,
            "lr": lr,
            "numEpochs": numEpochs,
            "z": z_size,
        },
    )

    modelVIB = MLPVIB()
    modelVIB.to(device)
    z_vib = modelVIB.fcmu.out_features
    optimizerVIB = optim.Adam(modelVIB.parameters(), lr=lr)

    historyVIB = trainRun(
        modelVIB,
        trainLoad,
        testLoad,
        optimizerVIB,
        device,
        numEpochs,
        beta=beta,
        tag="VIB",
        meta={
            "model": "MLPVIB",
            "beta": beta,
            "batch": batch,
            "lr": lr,
            "numEpochs": numEpochs,
            "z": z_vib,
        },
    )



if __name__ == "__main__":
    main()
