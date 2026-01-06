from deepthon.nn.layers import Sequential, Layer, Dropout, BatchNorm


def build_mlp(input_dim, architecture):
    layers = []
    in_dim = input_dim

    for block in architecture:
        if isinstance(block, list):
            out_dim = block[0]
            act = block[1] if len(block) > 1 else "linear"
            layers.append(Layer(in_dim, out_dim, activation=act))
            in_dim = out_dim

        elif block == "dropout":
            layers.append(Dropout(0.2))

        elif block == "batchnorm":
            layers.append(BatchNorm(in_dim))

    return Sequential(layers)


def build_model(cfg):
    mcfg = cfg["model"]
    return build_mlp(
        input_dim=mcfg["input_dim"],
        architecture=mcfg["architecture"]
    )
