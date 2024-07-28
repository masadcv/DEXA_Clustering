from functools import partial

import torchxrayvision as xrv

model_name_to_func = {
    "xrv_vae": partial(xrv.autoencoders.ResNetAE, weights="101-elastic"),
}
