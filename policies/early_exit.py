import torch


def should_exit(prob_k, pval_k, tau_k, alpha_k):
    # prob_k: (B,C) softmax probs, pval_k: (B,) conformal p-values
    maxp = prob_k.max(dim=1).values
    confident = maxp >= tau_k
    safe = pval_k >= (1 - alpha_k)
    return confident | safe