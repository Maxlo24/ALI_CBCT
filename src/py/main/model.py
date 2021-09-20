from monai.networks.nets import UNETR

def Create_UNETR(label_nbr,cropSize):

    model = UNETR(
        in_channels=1,
        out_channels=label_nbr,
        img_size=cropSize,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.1,
    )

    return model