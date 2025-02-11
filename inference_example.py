from datasets import get_benchmark_for_task
from scripts.transformer_prediction_interface.base import FairPFNClassifier
from scripts.tabular_metrics import causal_fairness_total_effect, roc_auc_score


task_type = "fairness_multiclass"
if "datasets_dict" not in locals():
    datasets_dict = {}

datasets_dict[f"valid_{task_type}"], df = get_benchmark_for_task(
    task_type,
    split="valid",
    max_samples=10000,
    return_as_lists=False,
    sel=False
)

dataset_map = {}
for dataset in datasets_dict[f"valid_{task_type}"]:
    dataset_map[dataset.name] = dataset

fairpfn = FairPFNClassifier()

dataset = dataset_map['Total_effect_lawschool']

train_ds, test_ds = dataset.generate_valid_split(n_splits=2)


fairpfn.fit(train_ds.x, train_ds.y)
y_pred = fairpfn.predict_proba(test_ds.x)

print("\n############ Calculating ATE and AUC ###############")
ate = causal_fairness_total_effect(target=test_ds.y, pred=y_pred, x=test_ds.x, prot_attr=test_ds.x[:, 0], name=test_ds.name, dowhy_data=test_ds.dowhy_data)
auc = roc_auc_score(test_ds.y, y_pred[:,1])
print(f'ATE: {round(ate, 3)}, AUC: {round(auc, 3)}')
