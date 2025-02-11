from dataclasses import dataclass, asdict
import dataclasses
import typing as tp
import torch

def compatability_fixes(model_state, config):
    """Fixes for compatability with old models"""
    return {k.replace(".step_module_layer", ""): v for k, v in model_state.items()}


@dataclass()
class Checkpoint:
    state_dict: tp.Dict[str, tp.Any]
    optimizer_state: tp.Dict[str, tp.Any]
    scaler_state: tp.Optional[tp.Dict[str, tp.Any]]

    config: tp.Dict[str, tp.Any]
    trained_epochs_until_now: int

    def save(self, path: str):
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        checkpoint: tp.Dict[str, tp.Any] = torch.load(path, map_location="cpu")
        # this is here for backwards compatibility
        if "trained_epochs_until_now" not in checkpoint:
            checkpoint["trained_epochs_until_now"] = checkpoint["config"][
                "trained_epochs_until_now"
            ]

        return cls(
            state_dict=compatability_fixes(
                checkpoint["state_dict"], checkpoint["config"]
            ),
            optimizer_state=checkpoint["optimizer_state"],
            config={
                **checkpoint["config"],
                "trained_epochs_until_now": checkpoint["trained_epochs_until_now"],
            },
            trained_epochs_until_now=checkpoint["trained_epochs_until_now"],
            scaler_state=checkpoint["scaler_state"]
            if "scaler_state" in checkpoint
            else checkpoint.get("scalar_state"),
        )


def load_model(
    path: str,
    device: str,
    verbose: bool = True,
    overwrite_config_keys: tp.Optional[tp.Dict[str, tp.Any]] = None,
):
    """
    Loads a model from a given path and filename.
    It returns a Transformer model and a config. This is ideal for low-level inference.
    If you want to continue training, it is recommended to go one level deeper and use `Checkpoint.load` directly.

    Args:
        path (str): Path to the model
        device (str): Device to load the model to
        verbose (bool): Whether to print the loaded config
    """
    checkpoint: Checkpoint = Checkpoint.load(path)

    if overwrite_config_keys is not None:
        checkpoint.config = {**checkpoint.config, **overwrite_config_keys}

    import pickle as pkl
    with open('artifacts/fairpfn_model.pkl', 'rb') as f:
        model = pkl.load(f)

    model.load_state_dict(checkpoint.state_dict)

    return model, checkpoint.config