"""
Will contain the changes needed to perform activation interventions using PCA from datasets.
"""

from typing import Tuple, Optional, Union, Callable, List
from jaxtyping import Int, Float
import torch
import torch.nn.functional

from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.utils import get_act_name

from algebraic_value_editing.prompt_utils import get_block_name

import dataset_svd.utils


class ActivationAdditionDataset:
    """Specifies a prompt (e.g. "Bob went") and a coefficient and a
    location in the model, with an `int` representing the block_num in the
    model. This comprises the information necessary to
    compute the rescaled activations for the prompt.
    """

    coeff: float
    act_name: str
    text_dataset: List[str]
    token_dataset: List[Int[torch.Tensor, "seq"]]
    from_dataset: bool
    use_all_activations: bool = False

    def __init__(
        self,
        coeff: float,
        act_name: Union[str, int],
        prompt: Optional[List[str]] = None,
        tokens: Optional[List[Int[torch.Tensor, "seq"]]] = None,
        from_dataset: bool = True,
        use_all_activations: bool = False,
        prompt_2: Optional[List[str]] = None,
        from_pca: bool = True,
        from_difference: bool = False,
    ):
        """Specifies a model location (`act_name`) from which to
        extract activations, which will then be multiplied by `coeff`.
        If `prompt` is specified, it will be used to compute the
        activations. If `tokens` is specified, it will be used to
        compute the activations. If neither or both are specified, an error will be raised.

        Args:
            `coeff  : The coefficient to multiply the activations by.
            `act_name`: The name of the activation location to use. If
            is an `int`, then it specifies the input activations to
            that block number.
            `prompt`: The prompt to use to compute the activations.
            `tokens`: The tokens to use to compute the activations.
        """
        assert (prompt is not None) ^ (
            tokens is not None
        ), "Must specify either prompt or tokens, but not both."

        self.coeff = coeff

        # Set the activation name
        if isinstance(act_name, int):
            self.act_name = get_block_name(block_num=act_name)
            self.location = self.act_name
        else:
            self.act_name = act_name

        # Set the tokens
        if tokens is not None:
            assert len(tokens.shape) == 1, "Tokens must be a 1D tensor."
            self.tokens = tokens
        else:
            self.prompt = prompt  # type: ignore (this is guaranteed to be str)

        # Set whether this is from a dataset
        self.from_dataset = from_dataset

        # Set whether to use all activations
        self.use_all_activations = use_all_activations

        # Set the second dataset (may be None)
        self.prompt_2 = prompt_2

        # Set whether this is from PCA
        self.from_pca = from_pca

        # Set whether this is from a difference
        self.from_difference = from_difference

    def __repr__(self) -> str:
        if hasattr(self, "prompt"):
            return f"ActivationAddition({self.prompt}, {self.coeff}, {self.act_name})"
        return (
            f"ActivationAddition({self.tokens}, {self.coeff}, {self.act_name})"
        )


def activation_principal_component(
    model: HookedTransformer, activation_addition: ActivationAdditionDataset
) -> Float[torch.Tensor, "batch pos d_model"]:
    """
    Return the principal component of the activations for the given
    `activation_addition` at some given layer of the model.
    """
    # Find the location we will look at
    location = activation_addition.location

    # Get the activations Tensor
    activations = dataset_svd.utils.dataset_activations_optimised_new(
        model=model,
        dataset=activation_addition.prompt,
        location=location,
        max_batch_size=2,
        use_all_activations=activation_addition.use_all_activations,
    ).cuda()

    # Find the average l2 norm of the rows of the activations
    average_l2_norm = torch.mean(torch.linalg.norm(activations, dim=1))

    # Do SVD
    _, _, V_H = dataset_svd.utils.SVD(activations)

    # Take the principal component
    principal_component = V_H[0]

    # This is essentially a 1-dim vector. Reshape it to be 3-dim,
    # for consistency with the other activations
    principal_component = principal_component.reshape(1, 1, -1)

    # Return the principal component scaled by the average l2 norm
    return average_l2_norm * principal_component


def get_dataset_activations(
    model: HookedTransformer, activation_addition: ActivationAdditionDataset
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Takes a `ActivationAddition` and returns the principal component of the activations
    of the dataset, for the appropriate `act_name`. Rescaling is done by running
    the model forward with the prompt and then multiplying the
    activations by the coefficient `activation_addition.coeff`.
    """
    # Get the principal component for the activations of this dataset at the relevant layer
    # TODO: Implement this method, decide if I want it to be a method
    principal_component = activation_principal_component(model, activation_addition)

    # Return cached activations times coefficient
    return activation_addition.coeff * principal_component


def get_dataset_activations_difference(
    model: HookedTransformer, activation_addition: ActivationAdditionDataset
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Takes a `ActivationAddition` and returns the difference between the centres of the 
    activations of the two relevant datasets, for the appropriate `act_name`.
    Rescaling is done by running the model forward with the prompt and then multiplying the
    activations by the coefficient `activation_addition.coeff`.
    """
    assert activation_addition.prompt_2 is not None, "Must specify a second prompt to use this method."

    target_dataset = activation_addition.prompt
    baseline_dataset = activation_addition.prompt_2
    location = activation_addition.location
    use_all_activations = activation_addition.use_all_activations

    feature_vector = dataset_svd.utils.find_activations_centre_diff(
        model=model,
        target_dataset=target_dataset,
        baseline_dataset=baseline_dataset,
        location=location,
        max_batch_size=2,
        use_all_activations=use_all_activations,
    ).reshape(1, 1, -1).cuda()

    return activation_addition.coeff * feature_vector
