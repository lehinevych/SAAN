# To model functions
from functools import partial
from transformer.model import transformer as transformer_main
from transformer.model import model_params
import tensorflow as tf


def model_fn_transformer(
    batch_sequence,
    is_traininig=False,
    transformer_params=model_params,
    NEG_INF=-1e9,
    soft_attention=True,
    component_wise=False,
):

    transformer = transformer_main.Transformer(transformer_params, is_traininig)
    # Mask which indicates whici element were zero padded and should be ignored
    zero_mask = tf.reduce_all(tf.equal(batch_sequence, 0), axis=-1)
    zero_mask_f = tf.cast(zero_mask, tf.float32) * NEG_INF

    # Transformer_PART
    with tf.name_scope("Transformer_encoder"):

        output = transformer(batch_sequence)

    # DNN_PART
    with tf.name_scope("Framewise_aggregator"):
        units = 512 if component_wise else 1
        weights = tf.layers.dense(output, units=units, reuse=tf.AUTO_REUSE)

        weights += zero_mask_f[..., None]

        dist_function = (
            partial(tf.nn.softmax, axis=1)
            if soft_attention
            else tf.contrib.sparsemax.sparsemax
        )

        if not component_wise:
            weights = tf.squeeze(weights, axis=-1)

        if not component_wise or soft_attention:
            normalized_across_components = dist_function(weights)
        else:
            normalized_across_components = tf.transpose(
                tf.map_fn(dist_function, tf.transpose(weights, (0, 2, 1))), (0, 2, 1)
            )
        if not component_wise:
            normalized_across_components = normalized_across_components[..., None]

        aggregated_embeddings = normalized_across_components * batch_sequence
        aggregated_embeddings = tf.reduce_sum(aggregated_embeddings, axis=1)

        if component_wise:
            normalized_across_components = tf.linalg.norm(
                normalized_across_components, axis=-1
            )[..., None]

    return aggregated_embeddings, normalized_across_components
