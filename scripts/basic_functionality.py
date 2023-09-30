""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """

# %%
from typing import List

import torch
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, utils, hook_utils
from algebraic_value_editing.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)

utils.enable_ipython_reload()

# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda:1")

# %%

for prompt in ("I love dogs", " wedding"):
    tokens: torch.Tensor = model.to_tokens(prompt)
    next_logits = model(tokens)
    # Greedily sample token at each position and then decode to string
    next_tokens = torch.argmax(next_logits, dim=-1)
    next_tokens_str = list(map(model.to_string, next_tokens[0]))

    print(f"For prompt tokenization {model.to_str_tokens(tokens)}")
    print(f"Greedy next tokens are {next_tokens_str}\n")

# Test the corresponding ActivationAddition
act_add = ActivationAddition(prompt=" wedding", coeff=1.0, act_name=6)
hook_fns_dict = hook_utils.hook_fns_from_activation_additions(
    model=model, activation_additions=[act_add]
)
try:
    for act_name, hook_fns in hook_fns_dict.items():
        for hook_fn in hook_fns:
            model.add_hook(act_name, hook_fn)

    tokens: torch.Tensor = model.to_tokens("I love dogs")
    output_logits = model(tokens)

    output_tokens = torch.argmax(output_logits, dim=-1)
    output_tokens_str = list(map(model.to_string, output_tokens[0]))
    print(f"For prompt tokenization {model.to_str_tokens(tokens)}")
    print(f"Wedding-modified tokens are {output_tokens_str}\n")
finally:
    model.remove_all_hook_fns()

# %%
# Get top 5 tokens
act_adds = [
    ActivationAddition(prompt="Love ", coeff=5.0, act_name=6),
    ActivationAddition(prompt="Hate", coeff=-5.0, act_name=6),
]
hook_fns = hook_utils.hook_fns_from_activation_additions(
    model=model, activation_additions=act_adds
)
try:
    # for act_name, hook_fn in hook_fns.items():
    #     model.add_hook(act_name, hook_fn)

    tokens: torch.Tensor = model.to_tokens("I hate you because")
    output_logits = model(tokens)

    output_tokens = torch.topk(output_logits, k=5, dim=-1).indices
    pos: int = 2
    output_tokens_str = list(map(model.to_string, output_tokens[0, pos]))
    print(f"For prompt tokenization {model.to_str_tokens(tokens)}")
    print(
        f"After modification, the 5 tokens at pos={pos} are {output_tokens_str}\n"
    )
finally:
    model.remove_all_hook_fns()

# %%
activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="Happy",
        prompt2=" ",
        coeff=2000,
        act_name=20,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt=(
        "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling"
    ),
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%
