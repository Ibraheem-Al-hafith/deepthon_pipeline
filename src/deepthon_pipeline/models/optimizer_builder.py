from deepthon.nn.optimizers import SGD, Adam

def build_optimizer(ocfg):

    if ocfg.name.lower() == "sgd":
        return SGD(lr=ocfg["lr"])

    if ocfg.name.lower() == "adam":
        return Adam(lr=ocfg.lr)

    raise ValueError("Unsupported optimizer")
