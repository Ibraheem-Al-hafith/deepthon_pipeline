from deepthon.src.deepthon.nn.layers import Sequential, Layer, Dropout, BatchNorm, get_activation
from .registry import register_model

# ---------------- Block mapper registery -----------------
BLOCK_MAPPER = {}
def register_block(name):
    def wrapper(fn):
        BLOCK_MAPPER[name.lower()] = fn
        return fn
    return wrapper

# ------- Block Implementation -------------

@register_block("layer")
def map_linear(token):
    """
    Formats:
    [in_dim, out_dim]
    [in_dim, out_dim, activation]
    """
    in_dim, out_dim, *rest = token
    activation = rest[0] if rest else None
    return [Layer(in_dim, out_dim, activation=activation)]

@register_block("dropout")
def map_dropout(token):
    """
    ["dropout",p]
    """
    p = token[1] if len(token)>0 else 0.2
    return [Dropout(p)]

@register_block("batchnorm")
def map_batchnorm(token):
    """
    ["batchnorm", num_features]
    """
    num_features = token[1]
    return [BatchNorm(num_features=num_features)]

# ------ Dispatcher ----------
def map_block(token):
    """route token to correct mapper"""
    if isinstance(token[0], str):
        key = token[0].lower()
        if key not in BLOCK_MAPPER:
            raise ValueError(f"Unknown architecture block: {token}")
        return BLOCK_MAPPER[key](token)
    
    return BLOCK_MAPPER['linear'](token)

# -------- build sequential model --------

def build_sequential_from_arch(arch):
    layers = []
    for block in arch:
        layers.extend(map_block(block))
    return Sequential(layers)

# ---------- Public Model Builder ----------

@register_model("sequential")
def build_model_from_config(cfg):
    """
    Expected YAML:

    model:
      name: sequential
      architecture:
        - [30, 64, "relu"]
        - ["batchnorm"]
        - [64, 32, "relu"]
        - ["dropout", 0.5]
        - [32, 1, "sigmoid"]
    """

    arch = cfg["model"]["architecture"]
    return build_sequential_from_arch(arch)

        