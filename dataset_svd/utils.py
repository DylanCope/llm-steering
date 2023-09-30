# import plotly.express as px
# import plotly.io as pio
# import plotly.graph_objects as go
# pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from fancy_einsum import einsum
from typing import List, Optional, Callable, Tuple, Union


from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

# Create some functions to distinguish between local and colab runs
def is_colab():
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    return IN_COLAB


def load_model(
    model_name: str    
) -> HookedTransformer:
    """
    Load a transformer lens model.
    Load it to gpu memory if running on colab, otherwise load it to cpu memory.
    """
    model = HookedTransformer.from_pretrained(model_name)
    if is_colab():
        model = model.cuda()
    else:
        model = model.cpu()
    return model


def SVD(matrix: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Compute the SVD of a matrix.
    Returns the three associated matrices
    """
    U, S, V_H = t.linalg.svd(matrix)
    return U, S, V_H


def dataset_activations(
    model: HookedTransformer,
    dataset: List[str]
):
    """
    Run a dataset through the model, store all the activations.
    Note: this is the unoptimised version, which runs the entire dataset through the model at once.
    This returns a cache with all activations (not just the final ones).
    """
    # Tokenise the batch, form a batch tensor
    batch_tokens = model.to_tokens(dataset)
    # Feed the tensor through the model
    logits, cache = model.run_with_cache(batch_tokens, return_cache_object=True, remove_batch_dim=False)

    return logits, cache


def dataset_activations_optimised(
    model: HookedTransformer,
    dataset: List[str],
    location: str,
    max_batch_size: int
):
    """
    Runs a dataset through the model, stores the activations of the final token of each sequence.
    This is the optimised version, which runs the dataset in batches.
    It also only stores the final activations, and not the entire cache.
    """
    num_batches = (len(dataset) + max_batch_size - 1) // max_batch_size
    all_final_activations = []
    # Process each batch
    for batch_idx in range(num_batches):
        t.cuda.empty_cache()
        # print("batch_idx be: ", batch_idx)

        # Determine the start and end index for this batch
        start_idx = batch_idx * max_batch_size
        end_idx = min(start_idx + max_batch_size, len(dataset))

        # Extract the subset of the dataset for this batch
        batch_subset = dataset[start_idx:end_idx]

        # Tokenise the batch, form a batch tensor
        batch_tokens = model.to_tokens(batch_subset)

        mask = batch_tokens != 50256
        final_indices = ((mask.cumsum(dim=1) == mask.sum(dim=1).unsqueeze(1)).int()).argmax(dim=1)
        final_indices = final_indices.view(-1,1)

        # Feed the tensor through the model
        # TODO: Use the names_filter argument to only return the activations we want
        _, cache = model.run_with_cache(batch_tokens, return_cache_object=True, remove_batch_dim=False)
        activations = cache[location]

        # # Take the last activation
        index_expanded = final_indices.unsqueeze(-1).expand(-1, -1, activations.size(2))
        # print("index_expanded: ", index_expanded)
        final_activations = t.gather(activations, 1, index_expanded)
        # Move the activations to the CPU and store them
        final_activations = final_activations.cpu()
        final_activations = final_activations.squeeze()
        all_final_activations.append(final_activations)

    # # Concatenate all activation tensors into a single tensor
    all_final_activations = t.cat(all_final_activations, dim=0)

    if is_colab():
        all_final_activations = all_final_activations.cuda()

    return all_final_activations

def select_vectors_parallel(indices, Y):
    """
    For each index, select every vector in Y up to and including the index.

    Parameters
    ----------
    indices : torch.Tensor
        Tensor of indices.
    Y : torch.Tensor
        Tensor of shape (batch, length, dimension).

    Returns
    -------
    torch.Tensor
        Stacked vectors.
    """
    assert len(indices.shape) == 1, "Indices tensor should be 1D"
    assert len(Y.shape) == 3, "Y tensor should be 3D"
    assert indices.shape[0] == Y.shape[0], "Batch size should match"

    # Create a mask of shape (batch, length)
    mask = t.arange(Y.shape[1]).expand(indices.shape[0], Y.shape[1]).to(Y.device) <= indices.unsqueeze(1)

    # Expand mask to match the shape of Y
    mask = mask.unsqueeze(2).expand_as(Y)

    # Apply mask and remove extra dimensions
    selected_vectors = Y[mask].view(-1, Y.shape[-1])

    return selected_vectors

def dataset_activations_optimised_new(
  model: HookedTransformer,
  dataset: List[str],
  location: str,
  max_batch_size: int,
  use_all_activations: bool = False
):
  """
  Note: this function has been updated to also return all activations, if we want it to do this

  """
  num_batches = (len(dataset) + max_batch_size - 1) // max_batch_size
  all_activations = []

  # Process each batch
  for batch_idx in range(num_batches):
    t.cuda.empty_cache()
    # print("batch_idx be: ", batch_idx)
    # Determine the start and end index for this batch
    start_idx = batch_idx * max_batch_size
    end_idx = min(start_idx + max_batch_size, len(dataset))

    # Extract the subset of the dataset for this batch
    batch_subset = dataset[start_idx:end_idx]

    # Tokenise the batch, form a batch tensor
    batch_tokens = model.to_tokens(batch_subset)

    mask = batch_tokens != 50256
    final_indices = ((mask.cumsum(dim=1) == mask.sum(dim=1).unsqueeze(1)).int()).argmax(dim=1)
    final_indices = final_indices.view(-1,1)

    # print(batch_tokens)
    # Feed the tensor through the model
    _, cache = model.run_with_cache(batch_tokens, return_cache_object=True, remove_batch_dim=False)
    activations = cache[location]
    if use_all_activations:
    #   print("final_indices are: ", final_indices)
    #   print("final indices shape be: ", final_indices.shape)
      output_activations = select_vectors_parallel(final_indices.squeeze(), activations).cpu()
    else:
      index_expanded = final_indices.unsqueeze(-1).expand(-1, -1, activations.size(2))
      # print("index_expanded: ", index_expanded)
      final_activations = t.gather(activations, 1, index_expanded)
      # Move the activations to the CPU and store them
      final_activations = final_activations.cpu()
      output_activations = final_activations.squeeze()

    all_activations.append(output_activations)

  all_activations = t.cat(all_activations, dim=0)

  return all_activations

def dataset_activations_optimised_locations(
    model: HookedTransformer,
    dataset: List[str],
    layers: int,
    location: str,
    max_batch_size: int
):
    """
    Same as earlier function, but returns activations for all locations.
    """

    num_batches = (len(dataset) + max_batch_size - 1) // max_batch_size
    all_final_activations = {}

    # Process each batch
    for batch_idx in range(num_batches):
        t.cuda.empty_cache()
        # print("batch_idx be: ", batch_idx)
        # Determine the start and end index for this batch
        start_idx = batch_idx * max_batch_size
        end_idx = min(start_idx + max_batch_size, len(dataset))

        # Extract the subset of the dataset for this batch
        batch_subset = dataset[start_idx:end_idx]

        # Tokenise the batch, form a batch tensor
        batch_tokens = model.to_tokens(batch_subset)

        mask = batch_tokens != 50256
        final_indices = ((mask.cumsum(dim=1) == mask.sum(dim=1).unsqueeze(1)).int()).argmax(dim=1)
        final_indices = final_indices.view(-1,1)

        # print(batch_tokens)
        # Feed the tensor through the model
        _, cache = model.run_with_cache(batch_tokens, return_cache_object=True, remove_batch_dim=False)

        for layer in range(layers):
            activations = cache[location.format(layer)]
            # # Take the last activation
            index_expanded = final_indices.unsqueeze(-1).expand(-1, -1, activations.size(2))
            # print("index_expanded: ", index_expanded)
            final_activations = t.gather(activations, 1, index_expanded)
            # Move the activations to the CPU and store them
            final_activations = final_activations.cpu()
            final_activations = final_activations.squeeze()
            all_final_activations.setdefault(layer, []).append(final_activations)



    for layer in range(layers):
        all_final_activations[layer] = t.cat(all_final_activations[layer], dim=0)


    return all_final_activations


def reshape_activations(
    batch_activations: t.Tensor
) -> t.Tensor:
    """
    Rearrange a pytorch tensor of shape (batch_size, num_tokens, num_features) 
    into a pytorch tensor of shape (batch_size * num_tokens, num_features).
    """
    squeezed_tensor = einops.rearrange(batch_activations, 'b tokens dim -> (b tokens) dim')
    return squeezed_tensor


def activation_SVD(
    model: HookedTransformer,
    dataset: List[str],
    location: str
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Given a model, a dataset, and a location in the model, 
    compute the SVD of the activations at that location.
    """
    _, cache = dataset_activations(model, dataset)
    activation_cache = cache[location]
    squeezed_activations = reshape_activations(activation_cache)
    U, S, V_H = SVD(squeezed_activations)
    return U, S, V_H


def dataset_activations_tokens(
    model: HookedTransformer,
    dataset_tokens: t.Tensor
):
    """
    Run a dataset which is already tokenised (and stored as a tensor)
    through the model, return the logits and all the activations.
    """
    # Tokenise the batch, form a batch tensor
    batch_tokens = dataset_tokens
    # Feed the tensor through the model
    logits, cache = model.run_with_cache(batch_tokens, return_cache_object=True, remove_batch_dim=False)
    return logits, cache


def activation_SVD_tokens(
    model: HookedTransformer,
    dataset_tokens: t.Tensor,
    location: str
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Given a model, a dataset (expressed as a tensor of tokens),
    and a location in the model, compute the SVD of the activations at that location.
    """
    _, cache = dataset_activations_tokens(model, dataset_tokens)
    activation_cache = cache[location]
    squeezed_activations = reshape_activations(activation_cache)
    U, S, V_H = SVD(squeezed_activations)
    return U, S, V_H


def activation_SVD_covariance(
    model: HookedTransformer,
    dataset_tokens: List[str],
    location: str
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Given a model, a dataset (in text), and a location in the model, 
    compute the SVD of the normalised covariance activation matrix at that location.
    """
    _, cache = dataset_activations_tokens(model, dataset_tokens)
    activation_cache = cache[location]
    squeezed_activations = reshape_activations(activation_cache)
    print(squeezed_activations.shape)
    mean_activation = squeezed_activations.mean(dim=0, keepdim=True)
    centred_activations = squeezed_activations - mean_activation
    covariance_matrix = centred_activations.T @ centred_activations
    print(covariance_matrix.shape)
    U, S, V_H = SVD(covariance_matrix)
    return U, S, V_H

# Using tokens as a starting point, and also using the covariance matrix
def activation_SVD_tokens_covariance(
    model: HookedTransformer,
    dataset_tokens: t.Tensor,
    location: str
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Given a model, a dataset (expressed as a tensor of tokens),
    and a location in the model, 
    compute the SVD of the normalised covariance activation matrix at that location.
    """
    _, cache = dataset_activations_tokens(model, dataset_tokens)
    activation_cache = cache[location]
    squeezed_activations = reshape_activations(activation_cache)
    mean_activation = squeezed_activations.mean(dim=0, keepdim=True)
    centred_activations = squeezed_activations - mean_activation
    covariance_matrix = centred_activations.T @ centred_activations
    U, S, V_H = SVD(covariance_matrix)
    return U, S, V_H

# Utils for comparing error between different SVDs

def dataset_projection(
    X: t.Tensor,
    B: t.Tensor
) -> t.Tensor:
    """
    Take in dataset X (with datapoints as rows) and an orthogonal basis B of a subspace.

    The basis should be of dimension D x M, with M vectors in the basis each of dim D.

    Compute the projection of each datapoint on this subspace, store the results as
    another dataset in the same form as the original.
    """

    return B.T @ B @ X.T


def top_k_projection(
    X: t.Tensor,
    V_H: t.Tensor,
    k: int
) -> t.Tensor:
    """
    Project the dataset X onto the top k orthogonal basis vectors in V_H
    """
    B = V_H[:k,:]
    proj = dataset_projection(X, B)
    return proj.T

def matrix_error(
    X_1: t.Tensor,
    X_2: t.Tensor
) -> int:
    """
    Treat X_1 and X_2 as rows of datapoints.
    Then, take the difference between these matrices,
    compute the l_2 norm of each row,
    and then sum these norms.

    Return the sum of these norms.
    """
    X = X_1 - X_2
    norms = t.norm(X, dim=1, p=2)
    sum_of_norms = t.sum(norms)

    return sum_of_norms

def find_activations_centre(
  model: HookedTransformer,
  dataset: List[str],
  location: str,
  max_batch_size: int,
  use_all_activations: bool = False
):
  """
  Find the centre of the activations of a dataset, at some
  layer of a certain model.
  """
  all_activations = dataset_activations_optimised_new(
    model,
    dataset,
    location,
    max_batch_size,
    use_all_activations
  )

  # Find the mean
  mean = t.mean(all_activations, dim=0)


  return mean

def find_activations_centre_diff(
  model: HookedTransformer,
  target_dataset: List[str],
  baseline_dataset: List[str],
  location: str,
  max_batch_size: int,
  use_all_activations: bool = False
):
  """
  Find the centre of the activations of the baseline dataset,
  take this away from the centre of the activations of a second dataset.

  Return the resulting difference vector
  """

  baseline_centre = find_activations_centre(
    model,
    baseline_dataset,
    location,
    max_batch_size,
    use_all_activations
  )

  baseline_target = find_activations_centre(
    model,
    target_dataset,
    location,
    max_batch_size,
    use_all_activations
  )

  difference = baseline_target - baseline_centre
  return difference