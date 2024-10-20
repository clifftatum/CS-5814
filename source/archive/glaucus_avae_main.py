

import torch

from glaucus import GlaucusRVQVAE

if __name__ == '__main__':
    # Pretrained Model (we will have to train one, linked some code below)
    model = GlaucusRVQVAE(quantize_dropout=True)
    # get weights (loading in one of The Areospace corp.'s pre-trained models)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://github.com/the-aerospace-corporation/glaucus/releases/download/v2.0.0/gvq-1024-a4baf001.pth",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.freeze()
    model.eval()

    # The model will take any (batch_size, length) as a complex tensor. The forward function will encode and decode sequentially returning the same shape as the input.
    x_tensor = torch.randn(11, 11113, dtype=torch.complex64)
    y_tensor = model(x_tensor)  # shape (11, 11113)

    # To get the compressed features from the input signal:
    x_tensor = torch.randn(3, 65536, dtype=torch.complex64)
    y_encoded, y_scale = model.encode(x_tensor)  # shapes ((3, 512, 16), (3,1))
    y_tensor_rms = model.decode(y_encoded)  # shape (3, 65536)
    y_tensor = model.decode(y_encoded, y_scale)  # shape (3, 65536)

    # The pretrained model has a base compression of 51.2x,
    # but can be scaled to 819.2x if desired by discarding N codebooks up to num_quantizers - 1.
    # This will reduce reconstruction accuracy:
    y_encoded_truncated = y_encoded[..., :9]  # keep 9 of 16 codebooks; new shape (3, 512, 9)
    y_tensor_57x = model.decode(y_encoded_truncated, y_scale)  # shape (3, 65536)

    pass






















# If getting the cbitstruct error, download this for windows OS https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Specifcall Desktop environment with C++ (MSVC)

# Conventionally Autoencoders use binary cross entropy loss
# https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-signal-noise-removal-autoencoder-with-keras.md
# https://machinecurve.com/index.php/2019/10/04/about-loss-and-loss-functions#binary-crossentropy

# We could use the glaucus loss:
# And add an attention layer; which would be in combination with the loss, a novel contribution
# https://medium.com/@williamjudge94/enhancing-autoencoders-with-attention-for-improved-anomaly-detection-85cab5c0969e

# Supporting material for autoencoders:
# https://medium.com/@sriskandaryan/autoencoders-demystified-audio-signal-denoising-32a491ab023a
    pass
    # loss = RFLoss(spatial_size=128, data_format='nl')
    #
    # # create some signal
    # xxx = torch.randn(128, dtype=torch.complex64)
    # # alter signal with 1% freq offset
    # yyy = xxx * np.exp(1j * 2 * np.pi * 0.01 * np.arange(128))

