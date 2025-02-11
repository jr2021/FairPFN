from __future__ import annotations

import copy
import pandas as pd
import tqdm
import os

import torch
import numpy as np
import openml
from pathlib import Path
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.sparse._csr import csr_matrix
from typing import Optional, Union, Literal, List
from aif360.sklearn.datasets import fetch_adult, fetch_bank, fetch_compas, fetch_german, fetch_lawschool_gpa, fetch_meps
from fairlearn.datasets import _fetch_boston, _fetch_credit_card, _fetch_diabetes_hospital, _fetch_acs_income
import dowhy.datasets
import pickle as pkl
from dowhy import CausalModel
from IPython.display import Image, display
from IPython.display import clear_output

from dowhy import CausalModel
import networkx as nx
import openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")


class DatasetModifications:
    def __init__(self, classes_capped: bool, feats_capped: bool, samples_capped: bool):
        """
        :param classes_capped: Whether the number of classes was capped
        :param feats_capped: Whether the number of features was capped
        :param samples_capped: Whether the number of samples was capped
        """
        self.classes_capped = classes_capped
        self.feats_capped = feats_capped
        self.samples_capped = samples_capped


class TabularDataset:
    def __init__(
        self,
        name: str,
        x: torch.tensor,
        y: torch.tensor,
        task_type: str,
        attribute_names: list[str],
        categorical_feats: Optional[list[int]] = None,
        modifications: Optional[DatasetModifications] = None,
        splits: Optional[list[tuple[torch.tensor, torch.tensor]]] = None,
        benchmark_name: Optional[str] = None, #TODO -Jake
        extra_info: Optional[dict] = None,
        description: Optional[str] = None,
    ):
        """
        :param name: Name of the dataset
        :param x: The data matrix
        :param y: The labels
        :param categorical_feats: A list of indices of categorical features
        :param attribute_names: A list of attribute names
        :param modifications: A DatasetModifications object
        :param splits: A list of splits, each split is a tuple of (train_indices, test_indices)
        """
        if categorical_feats is None:
            categorical_feats = []

        self.name = name
        self.x = x
        self.y = y
        self.categorical_feats = categorical_feats
        self.attribute_names = attribute_names
        self.modifications = (
            modifications
            if modifications is not None
            else DatasetModifications(
                classes_capped=False, feats_capped=False, samples_capped=False
            )
        )
        self.splits = splits
        self.task_type = task_type
        self.benchmark_name = benchmark_name
        self.extra_info = extra_info
        self.description = description

        if self.task_type in ("multiclass", "fairness_multiclass"):
            from model.encoders import MulticlassClassificationTargetEncoder

            self.y = MulticlassClassificationTargetEncoder.flatten_targets(self.y)

    def get_dataset_identifier(self):
        if self.task_type == "fairness_multiclass":
            return self.name
        else:
            tid = (
                    self.extra_info["openml_tid"]
                    if "openml_tid" in self.extra_info
                    else "notask"
                )
            did = (
                self.extra_info["openml_did"]
                if "openml_did" in self.extra_info
                else self.extra_info["did"]
            )
            return f"{did}_{tid}"

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(self.x.numpy(), columns=self.attribute_names)
        df = df.astype({name: "category" for name in self.categorical_names})
        df["target"] = self.y.numpy()
        return df

    @property
    def categorical_names(self) -> list[str]:
        return [self.attribute_names[i] for i in self.categorical_feats]

    def infer_and_set_categoricals(self) -> None:
        """
        Infers and sets categorical features from the data and sets the categorical_feats attribute. Don't use this
        method if the categorical indicators are already known from a predefined source. This method is used to infer
        categorical features from the data itself and only an approximation.
        """
        dummy_df = pd.DataFrame(self.x.numpy(), columns=self.attribute_names)
        encoded_with_categoricals = infer_categoricals(
            dummy_df,
            max_unique_values=20,
            max_percentage_of_all_values=0.1,
        )

        categorical_idxs = [
            i
            for i, dtype in enumerate(encoded_with_categoricals.dtypes)
            if dtype == "category"
        ]
        self.categorical_feats = categorical_idxs

    def __getitem__(self, indices):
        # convert a simple index x[y] to a tuple for consistency
        # if not isinstance(indices, tuple):
        #    indices = tuple(indices)
        ds = copy.deepcopy(self)
        ds.x = ds.x[indices]
        ds.y = ds.y[indices]

        self.indices = indices

        if self.task_type == "fairness_multiclass":
            ds.prot_attrs = ds.prot_attrs[indices]
            if ds.dowhy_data is not None:
                ds.dowhy_data['df'] = ds.dowhy_data['df'].iloc[indices]
            if 'counterfactual' in ds.name:
                ds.dowhy_data['df_cntf'] = ds.dowhy_data['df_cntf'].iloc[indices]

        return ds

    @staticmethod
    def check_is_valid_split(task_type, ds, index_train, index_test):
        if task_type not in ("multiclass", "fairness_multiclass"):
            return True

        # Checks if the set of classes are the same in dataset and its subsets
        if set(torch.unique(ds.y[index_train]).tolist()) != set(
            torch.unique(ds.y).tolist()
        ):
            return False
        if set(torch.unique(ds.y[index_test]).tolist()) != set(
            torch.unique(ds.y).tolist()
        ):
            return False

        return True

    def generate_valid_split(
        self,
        n_splits: int | None = None,
        splits: list[list[list[int], list[int]]] | None = None,
        split_number: int = 1,
        auto_fix_stratified_splits: bool = False,
    ) -> tuple[TabularDataset, TabularDataset] | tuple[None, None]:
        """Generates a deterministic train-(test/valid) split.

        Both splits must contain the same classes and all classes in the entire datasets.
        If no such split can be sampled, returns None.

        :param splits: A list of splits, each split is a tuple of (train_indices, test_indices) or None. If None, we generate the splits.
        :param n_splits: The number of splits to generate. Only required if splits is None.
        :param split_number: The split id. n_splits are coming from the same split and are disjoint. Further splits are
            generated by changing the seed. Only used if splits is None.
        :auto_fix_stratified_splits: If True, we try to fix the splits if they are not valid. Only used if splits is None.

        :return: the train and test split in format of TabularDataset or None, None if no valid split could be generated.
        """
        if split_number == 0:
            raise ValueError("Split number 0 is not used, we index starting from 1.")
        # We are using split numbers from 1 to 5 to legacy reasons
        split_number = split_number - 1

        if splits is None:
            if n_splits is None:
                raise ValueError("If `splits` is None, `n_splits` must be set.")
            # lazy import as not needed elsewhere.
            from utils import get_cv_split_for_data

            # assume torch tensor as nothing else possible according to typing.
            x = self.x if isinstance(self.x, np.ndarray) else self.x.numpy()
            y = self.y if isinstance(self.y, np.ndarray) else self.y.numpy()

            splits, *_ = get_cv_split_for_data(
                x=x,
                y=y,
                n_splits=n_splits,
                splits_seed=(split_number // n_splits)
                + 1,  # deterministic for all splits from one seed/split due to using //
                stratified_split=self.task_type in ("multiclass", "fairness_multiclass"),
                safety_shuffle=False,  # if ture, shuffle in the split function, and you have to update x and y
                auto_fix_stratified_splits=auto_fix_stratified_splits,
            )
            if isinstance(splits, str):
                print(f"Valid split could not be generated {self.name} due to {splits}")
                return None, None

            split_number_parsed = split_number % n_splits
            train_inds, test_inds = splits[split_number_parsed]
            train_ds = self[train_inds]
            test_ds = self[test_inds]
        else:
            train_inds, test_inds = splits[split_number]
            train_ds = self[train_inds]
            test_ds = self[test_inds]

        # print("dataset shape", self.x.shape)
        # print(self.x[:5])
        # print()

        train_ds.inds = train_inds
        test_ds.inds = test_inds

        return train_ds, test_ds

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def get_duplicated_samples(self, features_only=False, var_thresh=0.999999):
        """
        Calculates duplicated samples based on the covariance matrix of the data.

        :param features_only: Whether to only consider the features for the calculation
        :param var_thresh: The threshold for the variance to be considered a duplicate

        :return: Tuple of ((covariance matrix, duplicated_samples indices), fraction_duplicated_samples)
        """
        from utils import normalize_data

        if features_only:
            data = self.x.clone()
        else:
            data = torch.cat([self.x, self.y.unsqueeze(1)], dim=1)
        data[torch.isnan(data)] = 0.0

        x = normalize_data(data.transpose(1, 0))
        cov_mat = torch.cov(x.transpose(1, 0))
        cov_mat = torch.logical_or(cov_mat == 1.0, cov_mat > var_thresh).float()
        duplicated_samples = cov_mat.sum(axis=0)

        return (duplicated_samples, cov_mat), (
            duplicated_samples > 0
        ).float().mean().item()


def convert_categoricals(df):
    cat_columns = df.select_dtypes(["object", "category"]).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.astype("category"))
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def get_aif360_datasets(max_samples: int = 10000):
    fetch_funcs = [fetch_adult, fetch_bank, fetch_compas, fetch_german, fetch_lawschool_gpa, fetch_meps]
    
    aif360_datasets = []

    # Adult Income
    adult_dataset = fetch_adult(binary_race=True, dropcols=['race', 'sex'], dropna=True)
    X_adult, y_adult = adult_dataset.X, adult_dataset.y

    income_map = {'<=50K': 0, '>50K': 1}
    y_adult = y_adult.map(income_map)

    X_adult = X_adult.reset_index()

    race_map = {'Non-white': 0, 'White': 1}
    X_adult['race'] = X_adult['race'].map(race_map)
    sex_map = {'Female': 0, 'Male': 1}
    X_adult['sex'] = X_adult['sex'].map(sex_map)
    
    X_adult = convert_categoricals(X_adult)

    y_adult = y_adult.values[np.array(X_adult['race'].isna() == False)]
    X_adult = X_adult.dropna()
    attrs_adult = X_adult.columns

    if max_samples:
        X_adult, y_adult = X_adult.values[:max_samples], y_adult[:max_samples]

    aif360_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_adult).float(),
            y=torch.tensor(y_adult).float(),
            num_prot_attrs=1,
            attribute_names=attrs_adult,
            name='adult'
        )
    )
    
    # Bank Marketing
    bank_dataset = fetch_bank(dropcols=['age'], dropna=True)
    X_bank, y_bank = bank_dataset.X, bank_dataset.y
    
    subscription_map = {'yes': 1, 'no': 0}
    y_bank = y_bank.map(subscription_map)
    
    X_bank = X_bank.reset_index()

    X_bank['age'] = [0 if int(year) < 35 else 1 for year in X_bank['age']]

    X_bank = convert_categoricals(X_bank)
    attrs_bank = X_bank.columns
    
    if max_samples:
        X_bank, y_bank = X_bank.values[:max_samples], y_bank.values[:max_samples]

    aif360_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_bank).float(),
            y=torch.tensor(y_bank).float(),
            num_prot_attrs=1,
            attribute_names=attrs_bank,
            name='bank'
        )
    )

    # COMPAS Criminal Recidivism
    compas_dataset = fetch_compas(dropcols=['race', 'sex'], dropna=True, binary_race=True)
    X_compas, y_compas = compas_dataset.X, compas_dataset.y
    recid_map = {'Survived': 1, 'Recidivated': 0}
    y_compas = y_compas.map(recid_map)
    
    X_compas = X_compas.reset_index(level=0).reset_index(level=0)
    race_map = {'African-American': 0, 'Caucasian': 1}
    X_compas['race'] = X_compas['race'].map(race_map)
    sex_map = {'Female': 0, 'Male': 1} 
    X_compas['sex'] = X_compas['sex'].map(sex_map)
    
    X_compas = convert_categoricals(X_compas)
    attrs_compas = X_compas.columns

    if max_samples:
        X_compas, y_compas = X_compas.values[:max_samples], y_compas.values[:max_samples]

    aif360_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_compas).float(),
            y=torch.tensor(y_compas).float(),
            num_prot_attrs=1,
            attribute_names=attrs_compas,
            name='compas'
        )
    )

    # German Credit Risk
    german_dataset = fetch_german(binary_age=True, dropna=True, dropcols=['age', 'sex', 'foreign_worker'])
    X_german, y_german = german_dataset.X, german_dataset.y
    
    credit_map = {'good': 1, 'bad': 0}
    y_german = y_german.map(credit_map)
    
    X_german = X_german.reset_index(level=0).reset_index(level=1).reset_index(level=0)
    
    age_map = {'aged': 1, 'young': 0}
    X_german['age'] = X_german['age'].map(age_map)
    worker_map = {'yes': 0, 'no': 1}
    X_german['foreign_worker'] = X_german['foreign_worker'].map(worker_map)
    sex_map = {'female': 0, 'male': 1}
    X_german['sex'] = X_german['sex'].map(sex_map)

    X_german = convert_categoricals(X_german)
    attrs_german = X_german.columns

    if max_samples:
        X_german, y_german = X_german.values[:max_samples], y_german.values[:max_samples]

    aif360_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_german).float(),
            y=torch.tensor(y_german).float(),
            num_prot_attrs=1,
            attribute_names=attrs_german,
            name='german'
        )
    )

    # Medical Expenditure Panel Survey (MEPS)
    meps_dataset = fetch_meps(panel=19, accept_terms=True, dropna=True, dropcols=['RACE'])
    X_meps, y_meps = meps_dataset.X, meps_dataset.y

    util_map = {'< 10 Visits': 0, '>= 10 Visits': 1}
    y_meps = y_meps.map(util_map)

    X_meps = X_meps.reset_index()

    race_map = {'Non-White': 0, 'White': 1}
    X_meps['RACE'] = X_meps['RACE'].map(race_map)

    X_meps = convert_categoricals(X_meps)
    attrs_meps = X_meps.columns

    if max_samples:
        X_meps, y_meps = X_meps.values[:max_samples], y_meps.values[:max_samples]

    aif360_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_meps).float(),
            y=torch.tensor(y_meps).float(),
            num_prot_attrs=1,
            attribute_names=attrs_meps,
            name='meps'
        )
    )

    return aif360_datasets
    
def get_fairlearn_datasets(max_samples: int = 10000):
    fetch_funcs = [_fetch_boston, _fetch_credit_card, _fetch_diabetes_hospital, _fetch_acs_income]

    fairlearn_datasets = []

    X_credit, y_credit = _fetch_credit_card.fetch_credit_card(return_X_y=True)

    y_credit = [0 if default == '1' else 1 for default in y_credit]

    age_col = X_credit['x5']
    X_credit = X_credit.drop(labels=['x5'], axis=1)
    X_credit.insert(0, 'age', age_col)
    X_credit['age'] = [0 if year < 35 else 1 for year in X_credit['age']]

    sex_col = X_credit['x2']
    X_credit = X_credit.drop(labels=['x2'], axis=1)
    X_credit.insert(1, 'sex', sex_col)
    sex_map = {2: 0, 1: 1}
    X_credit['sex'] = X_credit['sex'].map(sex_map)

    X_credit = convert_categoricals(X_credit)

    if max_samples: 
        X_credit = X_credit.iloc[:max_samples]
        y_credit = y_credit[:max_samples]

    fairlearn_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_credit.values).float(),
            y=torch.tensor(y_credit).float(),
            num_prot_attrs=1,
            attribute_names=X_credit.columns,
            name='taiwan',
        )
    )

    # Diabetes 130-Hospitals
    X_diabetes, y_diabetes = _fetch_diabetes_hospital.fetch_diabetes_hospital(return_X_y=True)

    X_diabetes = X_diabetes.drop(labels=['readmitted', 'readmit_binary'], axis=1)

    race_map = {'Caucasian': 1, 'AfricanAmerican': 0, 'Asian': 0, 'Hispanic': 0, 'Other': 0, 'Unknown': 2}
    X_diabetes['race'] = X_diabetes['race'].map(race_map)

    gender_map = {'Male': 1, 'Female': 0, 'Unknown/Invalid': 2}
    X_diabetes['gender'] = X_diabetes['gender'].map(gender_map)

    age_map = {"'30 years or younger'": 1, "'30-60 years'": 1, "'Over 60 years'": 0}
    X_diabetes['age'] = X_diabetes['age'].map(age_map)

    y_diabetes = y_diabetes.values[(np.array(X_diabetes['race'] != 2)) & (np.array(X_diabetes['gender'] != 2))]    
    X_diabetes = X_diabetes[(np.array(X_diabetes['race'] != 2)) & (np.array(X_diabetes['gender'] != 2))]
            
    X_diabetes = convert_categoricals(X_diabetes)

    if max_samples: 
        X_diabetes = X_diabetes.iloc[:max_samples]
        y_diabetes = y_diabetes[:max_samples]

    fairlearn_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_diabetes.values).float(),
            y=torch.tensor(y_diabetes).float(),
            num_prot_attrs=1,
            attribute_names=X_diabetes.columns,
            name='diabetes',
        )
    )

    # American Community Survey (ACS) Income
    X_acs, y_acs = _fetch_acs_income.fetch_acs_income(return_X_y=True)

    X_acs, y_acs = X_acs[:max_samples], y_acs[:max_samples]

    X_acs[X_acs['RAC1P'] != 1.0] = 0.0
    X_acs[X_acs['SEX'] != 1.0] = 0.0
    X_acs['AGEP'] = [1.0 if age > X_acs['AGEP'].median() else 0.0 for age in X_acs['AGEP'].values]

    y_acs = [1.0 if income > y_acs.median() else 0.0 for income in y_acs.values]

    X_acs = X_acs[list(X_acs.columns[-2:]) + list(X_acs.columns[:-2])]

    if max_samples: 
        X_acs = X_acs.iloc[:max_samples]
        y_acs = y_acs[:max_samples]

    fairlearn_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_acs.values).float(),
            y=torch.tensor(y_acs).float(),
            num_prot_attrs=1,
            attribute_names=X_acs.columns,
            name='acs',
        )
    )

    # Boston Housing Dataset
    X_boston, y_boston = _fetch_boston.fetch_boston(return_X_y=True)

    X_boston['B'] = [1.0 if b > 136.9 else 0.0 for b in X_boston['B'].values]
    y_boston = [1.0 if y > y_boston.median() else 0.0 for y in y_boston]

    X_boston = X_boston[['B'] + list(X_boston.drop(labels=['B'], axis=1).columns)]

    if max_samples: 
        X_boston = X_boston.iloc[:max_samples]
        y_boston = y_boston[:max_samples]

    fairlearn_datasets.append(
        FairnessDataset(
            x=torch.tensor(X_boston.to_numpy(dtype=float)).float(),
            y=torch.tensor(y_boston).float(),
            num_prot_attrs=1,
            attribute_names=X_boston.columns,
            name='boston',
        )
    )

    return fairlearn_datasets

def get_communities_and_crime(max_samples: int = 10000):
    crime = openml.datasets.get_dataset(43888)
    X, _, _, _ = crime.get_data(dataset_format="dataframe")

    y = X['crimegt20pct']
    prot_col = X['blackgt6pct']
    X = X.drop(labels=['blackgt6pct'], axis=1)
    X.insert(0, 'blackgt6pct', prot_col)
    X.columns.values
    X = X.drop(columns=['V1', 'crimegt20pct', 'blackPerCap', 'fold'])

    X = convert_categoricals(X)

    if max_samples: 
        X = X.iloc[:max_samples]
        y = y.iloc[:max_samples]

    fairness_dataset = FairnessDataset(
        x=torch.tensor(X.to_numpy(dtype=float)).float(),
        y=torch.tensor(y).float(),
        num_prot_attrs=1,
        attribute_names=X.columns,
        name='crime'
    )

    return [fairness_dataset]

def load_causal_casestudies(n_max: int = 30, sel=True):
    causal_casestudies = ['Direct_effect', 'Indirect_effect', 'Indirect_effect_biased', 'Total_effect_level_one', 'Total_effect_level_two', 'Total_effect_level_three']
    cntf_casestudies = ['Direct_effect_counterfactual', 'Indirect_effect_counterfactual', 'Indirect_effect_biased_counterfactual',  'Total_effect_level_one_counterfactual', 'Total_effect_level_two_counterfactual', 'Total_effect_level_three_counterfactual']
    single_casestudies = ['Total_effect_lawschool', 'Total_effect_adult']
    single_casestudies_cntf = ['Total_effect_lawschool_counterfactual',  'Total_effect_adult_counterfactual']
    sel_casestudies = ['Total_effect_level_three', 'Total_effect_level_three_counterfactual']

    casestudy_benchmarks = []

    def get_pretty_name(casestudy):
        if 'level_three' in casestudy:
            return 'Fair Additive Noise'
        elif 'level_two' in casestudy:
            return 'Fair Unobservable'
        elif 'level_one' in casestudy:
            return 'Fair Observable'
        elif 'biased' in casestudy:
            return 'Biased'
        elif 'Direct_effect' in casestudy:
            return 'Direct-Effect'
        elif 'lawschool' in casestudy:
            return 'Law School Admissions'
        elif 'adult' in casestudy:
            return 'Adult Census Income'
        else:
            return 'Indirect-Effect'

    if sel_casestudies is not None and sel == True:
        for casestudy in sel_casestudies:
            data_path = Path('/work/dlclarge2/robertsj-fairpfn/prior-fitting/data/causal/casestudies') / casestudy
            for i in tqdm.tqdm(range(n_max), desc=f'{get_pretty_name(casestudy)} Benchmarks'):
                # try:
                with open(str(data_path) + f'/{casestudy}_{i}.pkl', 'rb') as f:
                    casestudy_benchmarks.append(pkl.load(f))
            #     except:
            # with open(str(data_path) + f'/{casestudy}.pkl', 'rb') as f:
            #     casestudy_benchmarks.append(pkl.load(f))
                    

        return casestudy_benchmarks

    print('\n########### Causal Case Studies ###########')

    for casestudy in causal_casestudies:
        data_path = Path('/work/dlclarge2/robertsj-fairpfn/prior-fitting/data/causal/casestudies') / casestudy
        for i in tqdm.tqdm(range(n_max), desc=f'{get_pretty_name(casestudy)} Benchmarks'):
            with open(str(data_path) + f'/{casestudy}_{i}.pkl', 'rb') as f:
                casestudy_benchmarks.append(pkl.load(f))
    
    print('\n########### Causal Case Studies (Counterfactual) ###########')

    for casestudy in cntf_casestudies:
        data_path = Path('/work/dlclarge2/robertsj-fairpfn/prior-fitting/data/causal/casestudies') / casestudy
        for i in tqdm.tqdm(range(n_max), desc=f'{get_pretty_name(casestudy)} Benchmarks'):
            with open(str(data_path) + f'/{casestudy}_{i}.pkl', 'rb') as f:
                casestudy_benchmarks.append(pkl.load(f))

    print('\n########### Real-World Data ###########')

    for casestudy in single_casestudies:
        print(f'{get_pretty_name(casestudy)}')
        data_path = Path('/work/dlclarge2/robertsj-fairpfn/prior-fitting/data/causal/casestudies') / casestudy
        with open(str(data_path) + f'/{casestudy}.pkl', 'rb') as f:
            casestudy_benchmarks.append(pkl.load(f))

    print('\n########### Real-World Data (Counterfactual) ###########')

    for casestudy in single_casestudies_cntf:
        print(f'{get_pretty_name(casestudy)}')
        data_path = Path('/work/dlclarge2/robertsj-fairpfn/prior-fitting/data/causal/casestudies') / casestudy
        with open(str(data_path) + f'/{casestudy}.pkl', 'rb') as f:
            casestudy_benchmarks.append(pkl.load(f))

    print('\n########### Loading FairPFN Model ###########')

    return casestudy_benchmarks

def get_benchmark_for_task(
    task_type: Literal["multiclass", "fairness_multiclass"],
    split: Literal["train", "valid", "debug", "test", "kaggle"] = "test",
    max_samples: Optional[int] = 10000,
    max_features: Optional[int] = 85,
    max_classes: Optional[int] = 2,
    max_num_cells: Optional[int] = None,
    min_samples: int = 50,
    filter_for_nan: bool = False,
    return_capped: bool = False,
    return_as_lists: bool = True,
    n_max: int = 200,
    load_data: bool = True,
    fairness_enabled: bool = False,
    sel=True,
) -> tuple[list[pd.DataFrame], pd.DataFrame | None]:
        
    if task_type == "fairness_multiclass":
        fairness_datasets = load_causal_casestudies(n_max=100, sel=sel)
        return fairness_datasets, None
    else:
        raise NotImplementedError(f"Unknown task type {task_type}")

    
class FairnessDataset(TabularDataset):
    def __init__(self, x: torch.tensor, y: torch.tensor, num_prot_attrs: int = 1, attribute_names: List[str] = [], name: str = "", dowhy_data: dict = None, **kwargs):
        """ """
        
        super().__init__(task_type="fairness_multiclass", x=x, y=y, attribute_names=attribute_names, name=name)
        
        self.num_prot_attrs = num_prot_attrs
        self.attribute_names = self.attribute_names
        self.prot_attrs = x[:, :num_prot_attrs]
        self.dowhy_data = dowhy_data

    def __repr__(self):
        return f"{self.name}"
    

class CausalFairnessDataset(FairnessDataset):
    def __init__(self, dowhy_data: dict, name: str, fair_observables: list = [], fair_unobservables: list = [], **kwargs):
        """ """
        x, y = dowhy_data['df'].drop(labels=['y'], axis=1), dowhy_data['df']['y']

        super().__init__(task_type="fairness_multiclass", 
                         x=torch.tensor(x.values).float(), 
                         y=torch.tensor(y.values).float(), 
                         num_prot_attrs=1,
                         attribute_names=x.columns, 
                         name=name, 
                         dowhy_data=dowhy_data,
                         **kwargs)
        
        self.dowhy_data = dowhy_data
        self.fair_observables = fair_observables
        self.fair_unobservables = fair_unobservables

    def __repr__(self):
        return f"{self.name}"

def infer_categoricals(
    df: pd.DataFrame,
    max_unique_values: int = 9,
    max_percentage_of_all_values: float = 0.1,
) -> pd.DataFrame:
    """
    Infers categorical features from the data and sets the categorical_feats attribute.
    :param df: Pandas dataframe
    :param max_unique_values: Maximum number of unique values for a feature to be considered categorical
    :param max_percentage_of_all_values: Maximum percentage of all values for a feature to be considered categorical
    :return: Pandas dataframe with categorical features encoded as category dtype
    """
    for column in df.columns:
        unique_values = df[column].nunique()
        unique_percentage = unique_values / len(df)

        if (
            unique_values <= max_unique_values
            and unique_percentage < max_percentage_of_all_values
        ):
            df[column] = df[column].astype("category")

    return df


def subsample(tensor, label_value, fraction):
    if label_value is None:
        num_samples = int(fraction * len(tensor))
        indices = torch.randperm(len(tensor))[:num_samples]
        return indices

    # Split tensor indices based on the label value
    matching_indices = (tensor[:, 0] == label_value).nonzero().squeeze()
    nonmatching_indices = (tensor[:, 0] != label_value).nonzero().squeeze()

    # Calculate how many matching rows we need to achieve the desired fraction
    num_matching = int(fraction * (len(nonmatching_indices) / (1.0 - fraction)))

    # If we need more matching rows than we have, adjust the number of non-matching rows
    if num_matching > len(matching_indices):
        num_matching = len(matching_indices)
        num_nonmatching = int(num_matching / fraction - num_matching)

        # Randomly select num_nonmatching rows
        indices_nonmatching = torch.randperm(len(nonmatching_indices))[:num_nonmatching]
        nonmatching_indices = nonmatching_indices[indices_nonmatching]

    # Randomly select num_matching rows
    indices_matching = torch.randperm(len(matching_indices))[:num_matching]
    selected_matching_indices = matching_indices[indices_matching]

    # Concatenate selected_matching_indices with nonmatching_indices
    result_indices = torch.cat((selected_matching_indices, nonmatching_indices), 0)

    # Shuffle the result tensor to avoid having all matching indices at the top
    result_indices = result_indices[torch.randperm(len(result_indices))]

    return result_indices


def remove_duplicated_datasets(dataset_list: List[TabularDataset]):
    """
    Removes datasets with duplicated names from the list of datasets.

    :param dataset_list: List
    :return:
    """
    seen = {}
    unique_objects = []
    for ds in dataset_list:
        if ds.name not in seen:
            unique_objects.append(ds)
            seen[ds.name] = True
    return unique_objects

