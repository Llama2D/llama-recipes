from dataclasses import dataclass


@dataclass
class llama2d_config:
    # use llama2d instead of llama
    use_2d:bool = False

    # when this is set to true, llama2d will ignore the positional embeddings. It should hopefully act just like llama.
    ignore_pos_embeds:bool = False