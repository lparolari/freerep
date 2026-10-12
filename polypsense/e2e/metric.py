import torch


def accuracy(logits, labels, topk=(1,)):
    """
    For each row in the logits matrix (query item), find the column (gallery item)
    with the maximum logit value (excluding the diagonal if it's self-similarity).
    Check if the predicted maximum corresponds to a ground truth positive label.

    logits: [b, b]
    labels: [b]
    """
    with torch.no_grad():
        device = logits.device

        # Create a binary mask for positive pairs
        targets = torch.eq(labels.view(-1, 1), labels.view(1, -1)).to(device)  # [b, b]

        # Mask diagonal (self-similarities): we don't want to count a match
        # between query and itself in gallery. For this reason we mask the
        # diagonal.
        n = logits.size(0)
        diag = torch.eye(n, dtype=bool).to(device)
        logits = logits.masked_fill(diag, float("-inf"))

        # Get top-k predictions as an index tensor where for each row (query) we
        # have the top-k indices of the gallery items.
        maxk = max(topk)
        _, preds = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [b, k]

        # Given that preds is an index to the top-k prediction for each query,
        # we construct a matrix such that for each i (query) we have k j
        # (predictions) where corrects[i, j] is True if the j-th prediction is
        # positive with respect to the query. Thus, correct [i, 0] is the top-1
        # prediction, if it is True then the top-1 prediction is positive for i.
        corrects = targets.gather(1, preds)  # [b, k]

        # Check if ground truth is in top-K
        accuracies = []
        for k in topk:
            corrects_k = corrects[:, :k].any(-1)  # [b]
            correct_k = corrects_k.float().sum()  # [1]
            accuracies.append(correct_k * 100.0 / n)
        return accuracies


def get_map_score(
    labels: torch.IntTensor, predictions: torch.IntTensor, g_labels: torch.IntTensor
):
    """
    From https://github.com/SerezD/information_retrieval_scores/blob/main/scores.py

    :param labels: (Q) query id labels, where Q is the length of query set
    :param predictions: (Q, G) of id labels, ranked from most likely.
                        Q = length of query set, G = length of gallery set
    :param g_labels: (G) gallery id labels, where G is the length of gallery set
    :return: mAP score

    Example:
    labels = [1, 4]
    g_labels = [1, 2, 3, 4, 5]
    predictions = [[2, 4, 1, 5, 3],[1, 3, 2, 5, 4]]

    Q = 2, G = 5

    AP_1 = (0 + 0 + 1/3 + 0 + 0) / 1 = 1/3
    AP_2 = (0 + 0 + 0 + 0 + 1/5) / 1 = 1/5
    mAP = (1/3 + 1/5) / 2 = 0.26667
    """

    # convert all to int 64
    labels = labels.type(torch.int64)
    g_labels = g_labels.type(torch.int64)
    predictions = predictions.type(torch.int64)

    g = g_labels.shape[0]
    device = labels.device

    # relevant docs in gallery for each label
    # (q)
    n_gtp = torch.sum((labels.unsqueeze(1) == g_labels), dim=1)

    # boolean mask on predictions
    # (q, g)
    mask = labels.unsqueeze(1) == predictions

    # number of retrieved documents at each position
    # (q, g)
    retrieved = torch.cumsum(mask, dim=1)

    # denominators (index if found retrieved, 0 otherwise)
    denominators = torch.arange(1, g + 1, device=device)
    denominators = torch.mul(mask, denominators)

    # precisions and average_precisions
    precisions = torch.nan_to_num(retrieved / denominators, nan=0, posinf=0, neginf=0)
    avg_precisions = torch.sum(precisions, dim=1)
    avg_precisions = avg_precisions / n_gtp

    mean_ap = torch.mean(avg_precisions)
    return mean_ap
