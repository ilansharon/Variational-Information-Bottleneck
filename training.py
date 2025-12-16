import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from losses import CrossEntropyLoss, VIBLoss, KL


def getLoaders(batch=128):
    transform = transforms.ToTensor()

    train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    trainLoad = DataLoader(train, batch_size=batch, shuffle=True)
    testLoad = DataLoader(test, batch_size=batch, shuffle=False)

    return trainLoad, testLoad


def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CorruptedMNIST(Dataset):
    def __init__(self, root, train=True, download=True, corruption_prob=0.0):
        self.mnist = datasets.MNIST(root=root, train=train, download=download, 
                                    transform=transforms.ToTensor())
        self.corruption_prob = corruption_prob
        self.targets = self.mnist.targets.clone()
        
        if corruption_prob > 0:
            n_samples = len(self.targets)
            n_corrupt = int(corruption_prob * n_samples)
            
            perm = torch.randperm(n_samples)[:n_corrupt]
            
            random_labels = torch.randint(0, 10, (n_corrupt,))
            self.targets[perm] = random_labels
            
            print(f"Corrupted {n_corrupt} labels out of {n_samples} ({corruption_prob*100:.1f}%)")

    def __getitem__(self, index):
        img, _ = self.mnist[index]
        target = self.targets[index]
        return img, target

    def __len__(self):
        return len(self.mnist)


def getCorruptedLoaders(corruption_prob=0.2, batch=128):
    train_ds = CorruptedMNIST(root="./data", train=True, corruption_prob=corruption_prob)
    trainLoad = DataLoader(train_ds, batch_size=batch, shuffle=True)
    
    test_ds = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
    testLoad = DataLoader(test_ds, batch_size=batch, shuffle=False)
    
    return trainLoad, testLoad


def setSeed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def baselineKLProxy(z):
    return 0.5 * (z ** 2).sum(dim=1).mean()


def oneEpoch(model, loader, optimizer, device):
    model.train()

    totalLoss = 0.0
    totalCE = 0.0
    totalKL = 0.0
    totalCorrect = 0
    totalExamples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, z = model(x)
        loss = CrossEntropyLoss(logits, y)
        kl = baselineKLProxy(z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch = x.size(0)
        totalLoss += loss.item() * batch
        totalCE += loss.item() * batch
        totalKL += kl.item() * batch

        preds = logits.argmax(dim=1)
        totalCorrect += (preds == y).sum().item()
        totalExamples += batch

    avgLoss = totalLoss / totalExamples
    avgCE = totalCE / totalExamples
    avgKL = totalKL / totalExamples
    accuracy = totalCorrect / totalExamples
    return avgLoss, avgCE, avgKL, accuracy


def oneEpochVIB(model, loader, optimizer, device, beta):
    model.train()

    totalLoss = 0.0
    totalCE = 0.0
    totalKL = 0.0
    totalCorrect = 0
    totalExamples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, mu, logvar, z = model(x)
        loss, ce, kl = VIBLoss(logits, y, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch = x.size(0)
        totalLoss += loss.item() * batch
        totalCE += ce.item() * batch
        totalKL += kl.item() * batch

        preds = logits.argmax(dim=1)
        totalCorrect += (preds == y).sum().item()
        totalExamples += batch

    avgLoss = totalLoss / totalExamples
    avgCE = totalCE / totalExamples
    avgKL = totalKL / totalExamples
    accuracy = totalCorrect / totalExamples
    return avgLoss, avgCE, avgKL, accuracy


def evaluate(model, loader, device):
    model.eval()

    totalLoss = 0.0
    totalCorrect = 0
    totalExamples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            logits = outputs[0]
            loss = CrossEntropyLoss(logits, y)

            batch = x.size(0)
            totalLoss += loss.item() * batch

            preds = logits.argmax(dim=1)
            totalCorrect += (preds == y).sum().item()
            totalExamples += batch

    avgLoss = totalLoss / totalExamples
    accuracy = totalCorrect / totalExamples
    return avgLoss, accuracy


def evaluateWithNoise(model, loader, device, sigma, num_samples=12):
    model.eval()
    totalCorrect = 0
    totalExamples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if sigma > 0:
                x = torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)

            sum_probs = torch.zeros(x.size(0), 10).to(device)
            
            try:
                for _ in range(num_samples):
                    outputs = model(x, force_sample=True) 
                    logits = outputs[0]
                    sum_probs += F.softmax(logits, dim=1)
                avg_probs = sum_probs / num_samples
                preds = avg_probs.argmax(dim=1)
                
            except TypeError: 
                outputs = model(x)
                logits = outputs[0]
                preds = logits.argmax(dim=1)

            batch = x.size(0)
            totalCorrect += (preds == y).sum().item()
            totalExamples += batch

    return totalCorrect / totalExamples

def trainRun(
    model,
    trainLoad,
    testLoad,
    optimizer,
    device,
    numEpochs,
    beta=None,
    tag="RUN",
    meta=None,
    log_every=1,
):
    history = {
        "trainLoss": [],
        "trainCE": [],
        "trainKL": [],
        "trainAcc": [],
        "testLoss": [],
        "testAcc": [],
        "meta": meta or {},
    }

    for epoch in range(1, numEpochs + 1):
        if beta is None:
            trainLoss, trainCE, trainKL, trainAcc = oneEpoch(model, trainLoad, optimizer, device)
            testLoss, testAcc = evaluate(model, testLoad, device)
        else:
            trainLoss, trainCE, trainKL, trainAcc = oneEpochVIB(model, trainLoad, optimizer, device, beta)
            testLoss, testAcc = evaluate(model, testLoad, device)

        should_log = log_every and (epoch % log_every == 0 or epoch == numEpochs)
        if should_log:
            print(
                f"[{tag}] Epoch {epoch:02d}: "
                f"train loss={trainLoss:.4f}, CE={trainCE:.4f}, KL={trainKL:.4f}, train acc={trainAcc:.4f}, "
                f"test loss={testLoss:.4f}, test acc={testAcc:.4f}"
            )

        history["trainLoss"].append(trainLoss)
        history["trainCE"].append(trainCE)
        history["trainKL"].append(trainKL)
        history["trainAcc"].append(trainAcc)
        history["testLoss"].append(testLoss)
        history["testAcc"].append(testAcc)

    return history
