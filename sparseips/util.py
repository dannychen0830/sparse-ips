import numpy as np
import jax.numpy as jnp


def pad_indices(ragged_list):
    """
    Converts a list of lists into a padded rectangular array and a mask.
    """
    num_rows = len(ragged_list)
    max_len = max((len(x) for x in ragged_list), default=0)

    # 1. Create the Map (Padded with 0)
    # We pad with 0 because it's a valid index. The mask handles the safety.
    padded_map = np.zeros((num_rows, max_len), dtype=np.int32)

    # 2. Create the Mask (1.0 for valid, 0.0 for padding)
    mask = np.zeros((num_rows, max_len), dtype=np.float32)

    for i, row in enumerate(ragged_list):
        length = len(row)
        padded_map[i, :length] = row
        mask[i, :length] = 1.0

    return jnp.array(padded_map), jnp.array(mask)
