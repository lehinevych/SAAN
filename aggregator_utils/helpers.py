import tensorflow as tf
import numpy as np


def pairwise_distances(embeddings, euclidian=True, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        euclidian: Boolean. If true, euclidian distance is calculated, cosine otherwise.
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    if not euclidian:
        norm = tf.reshape(tf.sqrt(square_norm), (-1, 1))
        norm_product = tf.matmul(norm, tf.transpose(norm))
        cosine_dist = tf.maximum(1 - tf.div(dot_product, norm_product), 0)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        cosine_dist = tf.maximum(cosine_dist, 0.0)
        return cosine_dist

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = (
        tf.expand_dims(square_norm, 0)
        - 2.0 * dot_product
        + tf.expand_dims(square_norm, 1)
    )

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def get_distance_statistics(label_mask, distances):

    """
    Input: tf.tensor nxn binary mask
           tf.tensor nxn cosine distance matrix
    Output: First and second moment of intra and inter class distance
    """

    diagonal = tf.zeros(tf.shape(label_mask)[0], dtype=tf.uint8)
    same_class_mask = tf.linalg.set_diag(label_mask, diagonal)
    same_class_ind = tf.where(tf.equal(same_class_mask, 1))
    diff_class_ind = tf.where(tf.equal(label_mask, 0))
    same_class_dist = tf.gather_nd(distances, same_class_ind)
    diff_class_dist = tf.gather_nd(distances, diff_class_ind)

    return (
        tf.nn.moments(same_class_dist, axes=0),
        tf.nn.moments(diff_class_dist, axes=0),
    )


def mean_aggregation(batch_embeddings):
    """
    Average pooling of the embeddings, ignoring zero padded ones.
    :param np.array batch_embeddings: batch_len x session_len x embedding_dim
    :return: np.array  batch_len x embedding_dim
    """
    zero_mask = tf.reduce_all(tf.equal(batch_embeddings, 0), axis=-1)
    zero_mask_int = tf.cast(zero_mask, tf.bool)
    num_elements = tf.count_nonzero(~zero_mask_int, axis=1)
    embeddings_mean = (
        tf.reduce_sum(batch_embeddings, axis=1)
        / tf.cast(num_elements, tf.float32)[..., None]
    )
    return embeddings_mean


def get_highest_lowest_attention_frames(attention_mask, zero_mask, POS_INFINITY=1e6):

    """
    :param attention_mask: n x n attention mask
    :param zero_mask: mask which indicates which embeddings were zero padded
    :param POS_INFINITY: value to mask zero_padedd embeddings
    :return: indecies of highest and lowest attention per session
    """
    zero_mask = tf.cast(zero_mask, tf.float32)
    max_ind = tf.cast(tf.argmax(attention_mask, axis=1), tf.int32)
    zero_mask *= POS_INFINITY
    attention_mask_biased = attention_mask + zero_mask[..., None]

    min_ind = tf.cast(tf.argmin(attention_mask_biased, axis=1), tf.int32)
    length = tf.shape(attention_mask)[0]
    row_nums = tf.range(length)[..., None]

    max_ind = tf.concat((row_nums, max_ind), axis=-1)
    min_ind = tf.concat((row_nums, min_ind), axis=-1)
    return max_ind, min_ind


def get_deviation_from_ave(attention_mask, zero_mask):

    """

    :param attention_mask: n x n attention mask
    :param zero_mask: mask which indicates which embeddings were zero padded
    :return: mean deaviation from average pooling
    """

    zero_mask_int = tf.cast(zero_mask, tf.bool)

    num_elements = tf.cast(tf.count_nonzero(~zero_mask_int, axis=1), tf.float32)
    mean_score = 1 / num_elements
    deviation = tf.abs(attention_mask - mean_score[..., None, None])

    deviation *= tf.cast(~zero_mask[..., None], tf.float32)
    mean_within_session = tf.reduce_sum(deviation, axis=1) / num_elements[..., None]

    mean_across_session = tf.reduce_mean(mean_within_session)
    return mean_across_session


def gen_label_mask(targets):
    """

    :param targets: nx1 ids of identities
    :return: nxn binary mask
    """
    v, h = tf.meshgrid(targets, targets)
    labels = tf.cast(tf.equal(v, h), tf.uint8)
    return labels


def tf_read_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    return image_decoded


def closest_class_prediction(pairwise_distances, labels):
    """
    Helper function to gather predictions for top-1 accuracy calculation

    :param pairwise_distances: nxn matrix with cosine distances within a batch
    :param labels: nx1 ids of identities
    :return:
    """

    max_values = tf.reduce_max(pairwise_distances)
    diag_replacer = tf.tile(
        tf.reduce_max(pairwise_distances)[None, ...], [tf.shape(pairwise_distances)[0]]
    )

    # The distance of embedding to itself will be 0, so we're replacing it with the max value
    replaced_diag = tf.linalg.set_diag(pairwise_distances, diag_replacer)

    indecies_min = tf.arg_min(replaced_diag, 1)

    predictions_raw = tf.gather(labels, indecies_min)

    # Filter classes with one instance only
    classes, _, counts = tf.unique_with_counts(labels)
    classes_not_for_acuracy = classes[tf.equal(counts, 1)]

    _, indecies_to_keep = tf.setdiff1d(labels, classes_not_for_acuracy)

    labels_selected = tf.gather(labels, indecies_to_keep)

    predictions = tf.gather(predictions_raw, indecies_to_keep)

    return predictions, labels_selected


def get_values_by_ohe(matrix, ohe_mask):

    indecies = tf.where(tf.cast(ohe_mask, tf.bool))
    values = tf.gather_nd(matrix, indecies)
    return values


def tf_deg2rad(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


def tf_rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180


def generate_flattened_label_mask(labels):
    v, h = np.meshgrid(labels, labels)
    mask = (v == h).astype(np.uint8)
    indecies = np.where(np.tri(len(labels), k=-1))
    return mask[indecies]


def flattened_vector_to_mask(flattened_vector, num_elements):
    initial_matrix = np.zeros((num_elements, num_elements))
    indecies = np.where(np.tri(num_elements, k=-1))
    initial_matrix[indecies] = flattened_vector
    initial_matrix += initial_matrix.T
    np.fill_diagonal(initial_matrix, 1)
    return initial_matrix


def tf_flattened_vector_to_mask(flattened_vector, num_elements):
    indecies = tf_generate_lower_triang_indecies(num_elements)
    lower_trian_matrix = tf.scatter_nd(
        indecies, flattened_vector, shape=(num_elements, num_elements)
    )
    diagonal = tf.ones(num_elements, dtype=flattened_vector.dtype)
    lower_trian_matrix += tf.transpose(lower_trian_matrix)
    final_matrix = tf.linalg.set_diag(lower_trian_matrix, diagonal)
    return final_matrix


def tf_generate_lower_triang_indecies(matrix_dim):
    indecies = tf.range(matrix_dim)

    indecies_stacked_v = tf.reshape(
        tf.tile(indecies, [matrix_dim]), (matrix_dim, matrix_dim)
    )

    indecies_stacked_h = tf.transpose(indecies_stacked_v)

    indecies = tf.concat(
        (
            tf.matrix_band_part(indecies_stacked_h, -1, 0)[..., None],
            tf.matrix_band_part(indecies_stacked_v, -1, 0)[..., None],
        ),
        axis=-1,
    )

    indecies = tf.reshape(indecies, (-1, 2))

    indecies_filtered = tf.squeeze(
        tf.gather(indecies, tf.where(tf.not_equal(indecies[..., 0], indecies[..., 1]))),
        axis=1,
    )

    tf.shape(indecies_filtered)

    return indecies_filtered


def create_unique_combs(sequence_elements):

    with tf.name_scope("unique_combinations"):
        element_len = tf.shape(sequence_elements)[0]

        lower_trian_indecies = tf_generate_lower_triang_indecies(element_len)

        combinations = tf.gather(sequence_elements, lower_trian_indecies)

    return combinations


def compute_cosine_sim(a, b):
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    sim = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return sim


def mask_condition(
    counter, length, indecies, quality_indecies, binary_mask, final_mask
):
    return tf.squeeze(counter) < length


def generate_row_col_index_fillers(indecies, matrix_dim):
    matrix_dim_vec = tf.reshape(matrix_dim, [1])

    vector_ind = tf.scatter_nd(
        indecies[..., None], tf.ones_like(indecies, dtype=tf.int64), matrix_dim_vec
    )

    matrix_dim_vec = tf.concat((tf.ones_like(matrix_dim_vec), matrix_dim_vec), axis=0)
    row_matrix = tf.cast(tf.tile(vector_ind[..., None], matrix_dim_vec), tf.bool)
    col_matrix = tf.transpose(row_matrix)
    final_matrix = tf.cast(tf.math.logical_or(row_matrix, col_matrix), tf.int64)
    return final_matrix


def matrix_true_cond_fun(non_zero_indecies, matrix):
    matrix_dim = tf.cast(tf.shape(matrix)[0], tf.int64)
    mask_to_apply = 1 - generate_row_col_index_fillers(non_zero_indecies, matrix_dim)
    return mask_to_apply * matrix


def body(counter, length, indecies, quality_indexes, binary_mask, final_mask):
    row_index_at_hand = tf.gather(quality_indexes, counter)

    row_at_hand = tf.squeeze(tf.gather(binary_mask, row_index_at_hand))

    matrix_update = tf.scatter_nd(
        tf.reshape(row_index_at_hand, (-1, 1)),
        row_at_hand[None, ...],
        tf.cast(tf.shape(final_mask), tf.int64),
    )
    final_mask += matrix_update

    non_zero_ind = tf.squeeze(tf.cast(tf.where(row_at_hand > 0), tf.int64))
    non_zero_ind = tf.reshape(non_zero_ind, (-1, 1))
    local_condition = tf.squeeze(tf.greater(tf.size(non_zero_ind), 0))
    index_true_cond_fun = lambda: tf.concat((indecies, row_index_at_hand), axis=0)
    result_ind = tf.cond(local_condition, index_true_cond_fun, lambda: indecies)
    result_matrix = tf.cond(
        local_condition,
        lambda: matrix_true_cond_fun(non_zero_ind, binary_mask),
        lambda: binary_mask,
    )

    return (counter + 1, length, result_ind, quality_indexes, result_matrix, final_mask)


def mask_postprocessing(binary_mask, quality_indexes):

    binary_mask = tf.cast(binary_mask, tf.int64)
    quality_indexes = tf.cast(quality_indexes, tf.int64)

    length = tf.cast((tf.shape(binary_mask)[0]), tf.int64)
    indecies = tf.zeros([0], dtype=tf.int64)
    counter = tf.constant([0], dtype=tf.int64)
    final_mask = tf.zeros_like(binary_mask)

    counter, length, result_ind, ranked_matrix, result_matrix, final_mask = tf.while_loop(
        mask_condition,
        body,
        loop_vars=[counter, length, indecies, quality_indexes, binary_mask, final_mask],
        shape_invariants=[
            counter.get_shape(),
            length.get_shape(),
            tf.TensorShape([None]),
            quality_indexes.get_shape(),
            binary_mask.get_shape(),
            final_mask.get_shape(),
        ],
        parallel_iterations=1,
    )

    return result_ind, final_mask


def tf_greedy_search_for_connected_components(binary_mask_full, attention_logits):
    quality_indecies = tf.argsort(tf.squeeze(attention_logits), direction="DESCENDING")

    unique_identities, final_mask = mask_postprocessing(
        binary_mask_full, quality_indecies
    )
    return unique_identities, final_mask


def convert_mask_to_face_tracks(embeddings, unique_binary_mask):
    unique_binary_mask = tf.cast(unique_binary_mask, tf.int32)
    mrange = tf.range(tf.shape(embeddings)[0]) + 1
    encoded = mrange * unique_binary_mask
    encoded = tf.sort(encoded, axis=1) - 1
    tf.cast(encoded, tf.int32)
    return tf.gather(embeddings, encoded)


def tf_rule_based_class_predictor(embeddings, distance_threshold=0.8):
    distances = 1 - compute_cosine_sim(embeddings, embeddings)
    indecies = tf_generate_lower_triang_indecies(tf.shape(embeddings)[0])
    distances_pairs = tf.gather_nd(distances, indecies)
    return tf.cast(distances_pairs < distance_threshold, tf.float32)


def componentwise_mask_mixing(gt_mask, pred_mask, is_training, num_training_steps=5000):
    if not is_training:
        return pred_mask, tf.constant(0)

    pred_mask = tf.cast(pred_mask, tf.int32)
    training_step = tf.train.get_global_step()

    probability_mix = tf.train.cosine_decay(
        1.0,
        training_step,
        num_training_steps,
        alpha=0.0,
        name="mix_component_probability",
    )

    bern_dist = tf.distributions.Bernoulli(probs=probability_mix)
    mask = bern_dist.sample(sample_shape=tf.shape(gt_mask))
    inverse_mask = tf.cast(1 - mask, tf.int32)
    final_mask = mask * gt_mask + inverse_mask * pred_mask
    return final_mask, probability_mix
