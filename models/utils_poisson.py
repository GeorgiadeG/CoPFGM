import torch
import numpy as np
import torch.nn.functional as F

def forward_pz(sde, config, samples_batch, m, labels):
    """Perturbing the augmented training data. See Algorithm 2 in PFGM paper.

    Args:
      sde: An `methods.SDE` object that represents the forward SDE.
      config: configurations
      samples_batch: A mini-batch of un-augmented training data
      m: A 1D torch tensor. The exponents of (1+\tau).

    Returns:
      Perturbed samples
    """
    tau = config.training.tau
    z = torch.randn((len(samples_batch), 1, 1, 1)).to(samples_batch.device) * config.model.sigma_end
    z = z.abs()
    # Confine the norms of perturbed data.
    # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
    if config.training.restrict_M:
        idx = (z < 0.005).squeeze()
        num = int(idx.int().sum())
        restrict_m = int(sde.M * 0.7)
        m[idx] = torch.rand((num,), device=samples_batch.device) * restrict_m

    data_dim = config.data.channels * config.data.image_size * config.data.image_size
    multiplier = (1+tau) ** m

    noise = torch.randn_like(samples_batch).reshape(len(samples_batch), -1) * config.model.sigma_end
    norm_m = torch.norm(noise, p=2, dim=1) * multiplier
    # Perturb z
    perturbed_z = z.squeeze() * multiplier
    # Sample uniform angle
    gaussian = torch.randn(len(samples_batch), data_dim).to(samples_batch.device)
    unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
    # Construct the perturbation for x
    perturbation_x = unit_gaussian * norm_m[:, None]
    perturbation_x = perturbation_x.view_as(samples_batch)
    # Perturb x
    perturbed_x = samples_batch + perturbation_x

    # num_classes = 10  # Total number of classes
    #
    # # Create the one-hot encoded labels tensor
    # one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
    #
    # # Replace the last 10 elements of each channel for each image with one_hot_labels
    # for j in range(perturbed_x.shape[0]):  # Loop over the batch size (images)
    #     for i in range(perturbed_x.shape[1]):  # Loop over the channels
    #         # Get the label of the jth image
    #         label = int(labels[j].item()) + 1
    #         ratio = float(label) / num_classes
    #         perturbed_x[j, i, :, :] = perturbed_x[j, i, :, :] * ratio

    # Augment the data with extra dimension z
    perturbed_samples_vec = torch.cat((perturbed_x.reshape(len(perturbed_x), -1),
                                       perturbed_z[:, None]), dim=1)

    return perturbed_samples_vec

# def forward_pz(sde, config, samples_batch, m, labels):
#     """Perturbing the augmented training data. See Algorithm 2 in PFGM paper.
#
#     Args:
#       sde: An `methods.SDE` object that represents the forward SDE.
#       config: configurations
#       samples_batch: A mini-batch of un-augmented training data
#       m: A 1D torch tensor. The exponents of (1+\tau).
#
#     Returns:
#       Perturbed samples
#     """
#     tau = config.training.tau
#     z = torch.randn((len(samples_batch), 1, 1, 1)).to(samples_batch.device) * config.model.sigma_end
#     z = z.abs()
#     # Confine the norms of perturbed data.
#     # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
#     if config.training.restrict_M:
#         idx = (z < 0.005).squeeze()
#         num = int(idx.int().sum())
#         restrict_m = int(sde.M * 0.7)
#         m[idx] = torch.rand((num,), device=samples_batch.device) * restrict_m
#
#     data_dim = config.data.channels * config.data.image_size * config.data.image_size
#     multiplier = (1+tau) ** m
#
#     noise = torch.randn_like(samples_batch).reshape(len(samples_batch), -1) * config.model.sigma_end
#     norm_m = torch.norm(noise, p=2, dim=1) * multiplier
#     # Perturb z
#     perturbed_z = z.squeeze() * multiplier
#     # Sample uniform angle
#     gaussian = torch.randn(len(samples_batch), data_dim).to(samples_batch.device)
#     unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
#     # Construct the perturbation for x
#     perturbation_x = unit_gaussian * norm_m[:, None]
#     perturbation_x = perturbation_x.view_as(samples_batch)
#     # Perturb x
#     perturbed_x = samples_batch + perturbation_x
#
#     num_classes = 10  # Total number of classes
#     one_hot_labels = F.one_hot(labels, num_classes=num_classes).unsqueeze(-1).unsqueeze(
#         -1).float()  # [batch_size, num_classes, 1, 1]
#
#     # Expand the one_hot_labels to match the dimension of the perturbed_x
#     one_hot_labels_expanded = one_hot_labels.expand(-1, -1, perturbed_x.shape[2],
#                                                     perturbed_x.shape[3])  # [batch_size, num_classes, width, height]
#
#     # Replace the last 10 elements of each channel for each image
#     perturbed_x[:, -num_classes:, :, :] = one_hot_labels_expanded
#
#     # Augment the data with extra dimension z
#     perturbed_samples_vec = torch.cat((perturbed_x,
#                                        perturbed_z[:, None]), dim=1)
#
#     return perturbed_samples_vec