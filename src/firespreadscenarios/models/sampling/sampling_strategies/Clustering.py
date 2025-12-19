from typing import Literal

import torch

from firespreadscenarios.models.sampling.generate_diverse import \
    edm_sampler_clustering
from firespreadscenarios.models.sampling.sampling_strategies.SamplingStrategy import \
    SamplingStrategy
from firespreadscenarios.models.utils import from_pm1_to_unit_interval
from firespreadscenarios.utils.hm_iou import compute_hm_iou


class Clustering(SamplingStrategy):
    """ """

    def __init__(
        self,
        clustering_distance_metric: Literal["chamfer", "L2"] = "chamfer",
        n_clusters: int = 8,
        prune_clusters: bool = True,
    ):
        self.clustering_distance_metric = clustering_distance_metric
        self.n_clusters = n_clusters
        self.prune_clusters = prune_clusters

    def sample(
        self, diffusion_model, latents, replicated_cond, existing_samples, hparams
    ):
        return edm_sampler_clustering(
            diffusion_model,
            latents,
            conditioning=replicated_cond,
            rho=hparams.rho,
            num_steps=hparams.timesteps,
            sigma_max=hparams.sigma_max,
            S_churn=hparams.s_churn,
            S_max=hparams.s_max,
            S_min=hparams.s_min,
            use_second_order_correction=hparams.use_second_order_correction,
            asymmetric_time_difference=hparams.asymmetric_time_difference,
            conditioning_noise_factor=hparams.conditioning_noise_factor,
            clustering_distance_metric=self.clustering_distance_metric,
            n_clusters=self.n_clusters,
            prune_clusters=self.prune_clusters,
        )

    def additional_metrics(self, sampling_results, batch_dict, i):
        x_pred = from_pm1_to_unit_interval(
            sampling_results["x_next"]
        )  # Size B x 1 x H x W
        binarized_predictions = (x_pred > 0.5).int().squeeze(1)
        possible_modes = batch_dict["all_targets"][i]

        stepwise_hm_ious = []
        additional_metrics_dict = {}
        if not self.prune_clusters:
            for cluster_ids in sampling_results["cluster_ids"]:
                if len(cluster_ids) == 0:
                    raise ValueError(
                        "No cluster ids found in the denoised samples. This should not happen."
                    )
                medoid_predictions = binarized_predictions[cluster_ids]
                deduped_iou = compute_hm_iou(
                    possible_modes.unique(dim=0), medoid_predictions
                )
                stepwise_hm_ious.append(deduped_iou)

            additional_metrics_dict = {
                "HM IoU of medoids after step " + str(k): stepwise_hm_ious[k].item()
                for k in range(len(stepwise_hm_ious))
            }

        return additional_metrics_dict

    def aggregate_additional_metrics(self, additional_metrics_agg):
        additional_metrics_agg_dict = {}
        for k in additional_metrics_agg[0].keys():
            additional_metrics_agg_dict[k] = (
                torch.tensor([d[k] for d in additional_metrics_agg])
                .float()
                .mean()
                .cpu()
                .item()
            )
        return additional_metrics_agg_dict
