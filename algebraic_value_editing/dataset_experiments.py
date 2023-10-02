from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    prompt_utils,
    metrics,
    sweeps,
    logits,
)

from algebraic_value_editing.dataset_utils import ActivationAdditionDataset

def run_corpus_logprob_experiment(
    model: HookedTransformer,
    labeled_texts: pd.DataFrame,
    dataset_vector: ActivationAdditionDataset,
    method: str = "mask_injection_logprob",
    text_col: str = "text",
    label_col: str = "label",
):
    """Function to evaluate log-prob on a set of input texts for both the
    original model and a model with various activation injections.  The
    injections are defined by a single pair of phrases and optional
    sweeps over coeff and act_name.  Results are presented over the
    classes present in the input text labels"""
    assert method in [
        "normal",
        "mask_injection_logprob",
        "pad",
    ], "Invalid method"

    # Create the metrics dict
    metrics_dict = {
        "logprob": metrics.get_logprob_metric(
            model,
            # agg_mode=["actual_next_token", "kl_div"],
            agg_mode=["actual_next_token"],
            # q_model=model,
            # q_funcs=(
            #     hook_utils.remove_and_return_hooks,
            #     hook_utils.add_hooks_from_dict,
            # ),
        )
    }

    activation_additions_df = pd.DataFrame([dataset_vector])
    
    # Create the texts to use, optionally including padding
    tokens_list = [model.to_tokens(text) for text in labeled_texts[text_col]]
    if method == "pad":
        activation_additions_all = [dataset_vector]
        tokens_list = [
            prompt_utils.pad_tokens_to_match_activation_additions(
                model=model,
                tokens=tokens,
                activation_additions=activation_additions_all,
            )[0]
            for tokens in tokens_list
        ]
    # Hack to avoid Pandas from trying to parse out the tokens tensors
    tokens_df = pd.DataFrame.from_records(
        [(tokens,) for tokens in tokens_list], index=labeled_texts.index
    ).rename({0: "tokens"}, axis="columns")
    # Get the logprobs on the original model
    normal_metrics = metrics.add_metric_cols(
        tokens_df,
        metrics_dict,
        cols_to_use="tokens",
        show_progress=True,
        prefix_cols=False,
    )
    # Get the modified model logprobs over all the ActivationAdditions
    mod_df = sweeps.sweep_over_metrics(
        model=model,
        inputs=tokens_df["tokens"],  # pylint: disable=unsubscriptable-object
        activation_additions=iter([dataset_vector]),
        metrics_dict=metrics_dict,
        prefix_cols=False,
    )
    # Join the normal logprobs into the patched df so we can take diffs
    mod_df = mod_df.join(
        normal_metrics[["logprob_actual_next_token"]],
        on="input_index",
        lsuffix="_mod",
        rsuffix="_norm",
    )

    # Add loss diff column
    mod_df["logprob_actual_next_token_diff"] = (
        mod_df["logprob_actual_next_token_mod"]
        - mod_df["logprob_actual_next_token_norm"]
    )
    # Create a loss sum column, optionally masking out the loss at
    # positions that had activations injected
    if method in ["mask_injection_logprob", "pad"]:
        # NOTE: this assumes that the same phrases are used for all
        # ActivationAdditions, which is currently the case, but may not always be!
        mask_pos = activation_additions_df.iloc[0]["activation_additions"][
            0
        ].tokens.shape[-1]
    else:
        mask_pos = 0
    mod_df["logprob_actual_next_token_diff_sum"] = mod_df[
        "logprob_actual_next_token_diff"
    ].apply(lambda inp: inp[mask_pos:].sum())
    # Create a token count column, so we can take the proper token mean
    # later. This count doesn't include any masked-out tokens (as it shouldn't)
    mod_df["logprob_actual_next_token_count"] = mod_df[
        "logprob_actual_next_token_diff"
    ].apply(lambda inp: inp[mask_pos:].shape[0])
    # Create a KL div mean column, also masking
    # TODO: fix this so we use the actual token mean, not
    # within-sentence mean then over-sentence mean.
    # mod_df["logprob_kl_div_mean"] = mod_df["logprob_kl_div"].apply(
    #     lambda inp: inp[mask_pos:].mean()
    # )
    # Group results by label, coeff and act_name, and take the sum
    results_grouped_df = (
        mod_df.groupby(["act_name", "coeff", label_col])
        .sum(numeric_only=True)
        .reset_index()
    )[
        [
            "act_name",
            "coeff",
            label_col,
            "logprob_actual_next_token_diff_sum",
            "logprob_actual_next_token_count",
        ]
    ]
    # Calculate the mean
    results_grouped_df["logprob_actual_next_token_diff_mean"] = (
        results_grouped_df["logprob_actual_next_token_diff_sum"]
        / results_grouped_df["logprob_actual_next_token_count"]
    )
    # Return the results
    return mod_df, results_grouped_df