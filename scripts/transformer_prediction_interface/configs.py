from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Any, Dict, List, Literal
import dataclasses
import math


@dataclass(eq=True, frozen=True)
class PreprocessorConfig:
    name: str
    categorical_name: Literal["none", "numeric", "onehot", "ordinal"] = "none"
    append_original: bool = True 

@dataclass
class TabPFNConfig:
    task_type: str
    model_type: Literal[
        "best", "single", "ensemble", "single_fast", "stacking", "bagging"
    ]
    paths_config: None
    task_type_config: None

    model_type_config: None = None

    model_name: str = "fairpfn"  # This name will be tracked on wandb

    preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
        PreprocessorConfig("safepower", categorical_name="numeric"),
        PreprocessorConfig("power", categorical_name="numeric"),
    )
    regression_y_preprocess_transforms: Tuple[str, ...] = (None,)
    feature_shift_decoder: Literal[
        "shuffle", "none", "local_shuffle", "rotate", "auto_rotate"
    ] = "shuffle"  # local_shuffle breaks, because no local configs are generated with high feature number
    normalize_with_test: bool = False
    fp16_inference: bool = True
    N_ensemble_configurations: int = 1
    average_logits: bool = True
    transformer_predict_kwargs: Optional[Dict] = field(default_factory=dict)
    save_peak_memory: Literal["True", "False", "auto"] = "True"
    batch_size_inference: int = None
    max_poly_features: int = 50
    use_poly_features: bool = True
    softmax_temperature: float = math.log(0.9)
    random_feature_scaling_strength: float = 0.0
    auxiliary_clf: str | None = None

    optimize_metric: Optional[str | None] = None
    c: Optional[Dict] = field(default_factory=dict)
    model: Optional[Any] = None

    def to_kwargs(self):
        kwargs = dataclasses.asdict(self)
        del kwargs["task_type"]
        del kwargs["model_type"]

        if self.task_type_config is not None:
            kwargs.update(dataclasses.asdict(self.task_type_config))

        if (
            kwargs.get("paths_config", None) is not None
            and kwargs.get("model", None) is not None
        ):
            raise ValueError(
                "Either paths_config or model must be specified, not both."
            )
        elif kwargs.get("paths_config", None) is not None:
            kwargs["model_string"] = kwargs["paths_config"]["model_strings"][0]
        elif kwargs.get("model", None) is not None:
            kwargs["model_string"] = "tabpfn"
        else:
            raise ValueError(
                f"Either paths_c"
                f"onfig or model must be specified paths_config {kwargs.get('paths_config', None)} model {kwargs.get('model', None)}"
            )

        del kwargs["paths_config"]
        del kwargs["task_type_config"]
        del kwargs["model_type_config"]
        del kwargs["model_name"]

        return kwargs
    
@dataclass
class TabPFNModelPathsConfig:
    paths: list[str]

    model_strings: list[str] = dataclasses.field(init=False)

    task_type: str = "fairness_multiclass"



    def __post_init__(self):
        # Initialize Model paths
        self.model_strings = []

        self.paths = ["/work/dlclarge2/robertsj-fairpfn/prior-fitting/results/models_diff/model_submitit_0c_id_e3e764c1_epoch_-1.cpkt"]

        for path in self.paths:
            self.model_strings.append(path)

@dataclass
class TabPFNClassificationConfig:
    multiclass_decoder: Literal[
        "shuffle", "none", "local_shuffle", "rotate"
    ] = "shuffle"


def get_params_from_config(c):
    return (
        {}  # here you add things that you want to use from the config to do inference in transformer_predict
    )