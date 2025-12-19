import torch
from torchmetrics.classification import BinaryJaccardIndex


def compute_ged(true_imgs: torch.tensor, gen_imgs: torch.tensor):
    # Computes the generalized energy distance between two probability distributions, based on the Jaccard index as the distance function d.

    # Threshold the images
    true_imgs = (true_imgs > 0.5).float().cuda()
    gen_imgs = (gen_imgs > 0.5).float().cuda()

    def compute_mean_distance(imgs1, imgs2):
        # Distance is 1-IoU(x,y)
        d = BinaryJaccardIndex().to("cuda")
        # torchmetrics's compute() would give us the overall IoU, but we want the mean per-image IoU
        ious = []
        for img1 in imgs1:
            for img2 in imgs2:
                ious.append(d(img1, img2))
        return 1 - torch.tensor(ious).nan_to_num(1.0).mean()

    def compute_mean_distance_within(imgs):
        # Use to compute the expected distance within the same set of images
        # Distance is 1-IoU(x,y)
        d = BinaryJaccardIndex().to("cuda")
        ious = []
        # Compute upper triangular matrix of distances
        for i in range(imgs.shape[0]):
            for j in range(i + 1, imgs.shape[0]):
                ious.append(d(imgs[i], imgs[j]))
        # Duplicate each entry, to account for the lower triangular matrix
        ious = ious * 2

        # Compute on-diagonal distances
        for i in range(imgs.shape[0]):
            ious.append(d(imgs[i], imgs[i]))

        return 1 - torch.tensor(ious).nan_to_num(1.0).mean()

    dist_true_gen = compute_mean_distance(true_imgs, gen_imgs)
    dist_true_true = compute_mean_distance_within(true_imgs)
    dist_gen_gen = compute_mean_distance_within(gen_imgs)

    return {
        "dist_true_gen": dist_true_gen,
        "dist_true_true": dist_true_true,
        "dist_gen_gen": dist_gen_gen,
        "ged": 2 * dist_true_gen - dist_true_true - dist_gen_gen,
    }
