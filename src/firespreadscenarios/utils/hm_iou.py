import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics.classification import BinaryJaccardIndex


def compute_iou_matrix(outputs, targets):
    # Zero division should only occur if both prediction and target contain no positive class,
    # which would represent a perfect match, so we set the IoU to 1 in that case.
    iou = BinaryJaccardIndex(zero_division=1.0).to(outputs.device)

    m = outputs.shape[0]
    n = targets.shape[0]

    iou_matrix = torch.zeros((m, n), device=outputs.device)

    for b1 in range(m):
        for b2 in range(n):
            iou_matrix[b1, b2] = iou(outputs[b1], targets[b2]).nan_to_num(1.0)
    return iou_matrix


def match_targets(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return row_ind, col_ind


def compute_hm_iou(
    true_imgs: torch.tensor, gen_imgs: torch.tensor, return_ious: bool = False
):
    iou_matrix = compute_iou_matrix(true_imgs, gen_imgs)
    row_ind, col_ind = match_targets(1 - iou_matrix)  # need cost matrix, so 1-IoU
    ious = iou_matrix[row_ind, col_ind]
    if return_ious:
        return ious.mean(), ious
    else:
        return ious.mean()
