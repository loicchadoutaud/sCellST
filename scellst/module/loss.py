import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import BCEWithLogitsLoss
from torchvision.ops import sigmoid_focal_loss


def mse(output: Tensor, target: Tensor) -> Tensor:
    return ((output - target) ** 2).mean()


def mae(output: Tensor, target: Tensor) -> Tensor:
    return torch.nn.functional.l1_loss(output, target)


def negative_log_likelihood(output: Distribution, target: Tensor) -> Tensor:
    return -output.log_prob(target.int()).sum(-1).mean()


def bce(
    output: Tensor, target: Tensor, weight: Tensor, epsilon: float = 1.0e-12
) -> Tensor:
    mask = torch.where(target == 0.5, 0., 1.).to(target.device)
    output = torch.clamp(output, min=epsilon, max=1 - epsilon)
    unreduced_losses = torch.nn.functional.binary_cross_entropy(
        output, target, weight=weight, reduction="none"
    )
    assert not torch.any(
        torch.isnan(unreduced_losses)
    ), "Nan present after loss computations"
    reduced_losses = (mask * unreduced_losses).sum(dim=0)
    n_samples = mask.sum(dim=0)
    n_samples = torch.where(n_samples == 0, 1, n_samples).to(target.device)
    loss = (reduced_losses / n_samples).mean()  # To avoid division by zero
    assert not torch.isnan(loss), "Nan present after loss reductions"
    return loss


def focal_loss_logit(output: Tensor, target: Tensor) -> Tensor:
    return sigmoid_focal_loss(output, target, reduction="mean")


def cosine(output: Tensor, target: Tensor) -> Tensor:
    return 1 - torch.nn.CosineSimilarity(dim=0)(output, target).mean()


def ranknet(y_pred: Tensor, y_true: Tensor, weight_by_diff: bool=False, weight_by_diff_powed: bool=False):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    pairwise_true_differences = y_true.unsqueeze(1) - y_true
    pairwise_true_differences = (pairwise_true_differences > torch.tensor(0., device=pairwise_true_differences.device)).float()

    pairwise_pred_difference = y_pred.unsqueeze(1) - y_pred

    weight = None
    if weight_by_diff:
        weight = torch.abs(pairwise_true_differences)
    elif weight_by_diff_powed:
        weight = pairwise_true_differences.pow(2)

    return BCEWithLogitsLoss(weight=weight)(pairwise_pred_difference, pairwise_true_differences)


# def soft_sort(y_pred: Tensor, y_true: Tensor,)


loss_dict = {"mse": mse, "mae": mae, "nll": negative_log_likelihood, "bce": bce, "cosine": cosine, "ranknet": ranknet}
