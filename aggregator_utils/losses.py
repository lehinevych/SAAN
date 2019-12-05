import tensorflow as tf
from aggregator_utils.helpers import *
import math


def get_arcface_logits(embedding, labels, out_num, w_init=None, s=64.0, m=0.35):
    """
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly;
            intra class angles,
            inter class angles
    """
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope("arcface_loss"):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name="norm_embedding")

        weights = tf.get_variable(
            name="embedding_weights",
            shape=(embedding.get_shape().as_list()[-1], out_num),
            initializer=w_init,
            dtype=tf.float32,
        )

        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name="norm_weights")
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name="cos_t")
        mask = tf.one_hot(labels, depth=out_num, name="one_hot_mask")

        positive_angles = tf_rad2deg(tf.acos(get_values_by_ohe(cos_t, mask)))
        negative_angles = tf.reshape(
            tf_rad2deg(tf.acos(get_values_by_ohe(cos_t, tf.abs(1 - mask)))), (1, -1)
        )

        cos_t2 = tf.square(cos_t, name="cos_2")
        sin_t2 = tf.subtract(1.0, cos_t2, name="sin_2")
        sin_t = tf.sqrt(sin_t2, name="sin_t")

        cos_mt = s * tf.subtract(
            tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name="cos_mt"
        )

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name="if_else"), dtype=tf.bool)

        keep_val = s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1.0, mask, name="inverse_mask")

        s_cos_t = tf.multiply(s, cos_t, name="scalar_cos_t")

        output = tf.add(
            tf.multiply(s_cos_t, inv_mask),
            tf.multiply(cos_mt_temp, mask),
            name="arcface_loss_output",
        )

    return output, positive_angles, negative_angles


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def batch_hard_triplet_loss(
    labels, embeddings, margin=None, squared=False, is_soft=False, is_euclidian=False
):

    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
        is_soft: Boolean. If true, use ln(1+e^(distance)) loss. No margin is needed then.  
        is_euclidian: Boolean. If true, euclidian distance is used as a margin. Cosine otherwise.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = pairwise_distances(
        embeddings, squared=squared, euclidian=is_euclidian
    )

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
        1.0 - mask_anchor_negative
    )

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    if is_soft:
        triplet_loss = tf.log(1 + tf.exp(hardest_positive_dist - hardest_negative_dist))
    else:
        triplet_loss = tf.maximum(
            hardest_positive_dist - hardest_negative_dist + margin, 0.0
        )

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss, anchor_positive_dist, anchor_negative_dist, pairwise_dist
