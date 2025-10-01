# configuration_omega.py

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OmegaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmegaModel`]. It is used to instantiate a
    Omega model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Omega model.
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key-value heads for GQA.
        head_dim (`int`, *optional*, defaults to 256):
            Dimension of attention head.
        hidden_activation (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The activation function in the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_local_base_freq (`float`, *optional*, defaults to 1000000.0):
            Local RoPE base frequency for sliding attention layers.
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        layer_types (`list`, *optional*):
            Types of layers (full_attention or sliding_attention).
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window size for local attention.
        query_pre_attn_scalar (`int`, *optional*, defaults to 224):
            Scalar for query pre-attention.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0):
            Attention logit softcapping value.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            Final logit softcapping value.
    """

    model_type = "omega"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_local_base_freq=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=None,
        sliding_window=4096,
        query_pre_attn_scalar=224,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ):
        # Initialize layer types if not provided
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types
        self.sliding_window = sliding_window
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Alias for backward compatibility
OmegaTextConfig = OmegaConfig