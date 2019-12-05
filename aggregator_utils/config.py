from collections import defaultdict

# params for single-identity batching
SINGLE_SAMPLER_PARAMS = defaultdict(
    lambda: None,
    num_people_to_sample=128,
    num_random_sessions=256,
    num_frames_per_session=32,
    prefetch_buffer=128,
    sample_frames_discretely=False,
    are_records_sorted=False,
)

# params for multi-identity batching
MULTI_SAMPLER_PARAMS = defaultdict(
    generated_session_len=256, identities_limit=16, prefetch_buffer=16
)


BASE_MODEL_PARAMS = defaultdict(
    lambda: None,
    arcface_radius=16.0,
    arcface_margin=0.35,
    component_wise_attention=False,
    sparse_attention=True,
    encoder_num_hidden_layers=4,
    encoder_self_attention_dropout=0.3,
    encoder_relu_dropout=0.4,
    encoder_return_attention_scores=False,
    encoder_use_positional_encoding=True,
)

# single-identity aggregator on ResNet features
RSAAN_PARAMS = BASE_MODEL_PARAMS.copy()
RSAAN_PARAMS.update(embeddings_dim=512, encoder_hidden_size=512, encoder_num_heads=8)

# single-identity aggregator on MobileNet features
MSAAN_PARAMS = BASE_MODEL_PARAMS.copy()
MSAAN_PARAMS.update(embeddings_dim=128, encoder_hidden_size=128, encoder_num_heads=4)

# multi-identity aggregator on ResNet features
MIRSAAN_PARAMS = RSAAN_PARAMS.copy()
MIRSAAN_PARAMS.update(
    binarization_threshold=0.8, num_iterations_for_teacher_forcing=5000
)
