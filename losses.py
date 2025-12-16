import torch
import torch.nn.functional as F

def CrossEntropyLoss(logits, targets):
    return F.cross_entropy(logits, targets)

def KL(mu, logvar):
    #closed form - 1 + logvar - mu^2 - var (from VIB paper)

    perDim = 1 + logvar - mu**2 - torch.exp(logvar)
    perSample = -0.5 * perDim.sum(dim=1)
    return perSample.mean()

def VIBLoss(logits, targets, mu, logvar, beta):
    ce = CrossEntropyLoss(logits, targets)
    kl = KL(mu, logvar)
    loss = ce + beta * kl
    return loss, ce, kl 