import numpy as np
from dowhy import CausalModel


def fairness_ddsp(target=None, pred=None, x=None, prot_attr=None, dowhy_data=None, name=None):
    return np.abs(np.mean(pred[:,1][prot_attr[:, 0] == 0.0]) - np.mean(pred[:,1][prot_attr[:, 0] == 1.0]))

def causal_fairness_total_effect(target=None, pred=None, x=None, prot_attr=None, dowhy_data=None, name=None):
    if "Total" not in name:
        return 0
    
    dowhy_data['df'] = dowhy_data['df'].iloc[:pred.shape[0]]

    dowhy_data['df'][dowhy_data['df'].columns[:-1]] = x
    dowhy_data['df']['y'] = pred[:,1]
    
    model = CausalModel(data=dowhy_data["df"],
                        treatment=dowhy_data["common_causes_names"], outcome=dowhy_data["outcome_name"],
                        graph=dowhy_data["gml_graph"])
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    linear_estimate = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=0,
                                            treatment_value=1,
                                            method_params={'need_conditional_estimates': False},
                                            evaluate_effect_strength=False)
    
    return linear_estimate.value

'''
    print("pred.shape", pred.shape)
    print("df['df'].shape", dowhy_data['df'].shape)
    print("df['df'].columns", dowhy_data['df'].columns)
    print("x.shape", x.shape)
'''
    
def causal_fairness_direct_effect(target=None, pred=None, x=None, prot_attr=None, dowhy_data=None, name=None):
    if "Direct" not in name:
        return 0

    dowhy_data['df'] = dowhy_data['df'].iloc[:pred.shape[0]]
    dowhy_data['df'][dowhy_data['df'].columns[:-1]] = x
    dowhy_data['df']['y'] = pred[:,1]
    
    model = CausalModel(data=dowhy_data["df"],
                        treatment=dowhy_data["effect_modifier_names"], outcome=dowhy_data["outcome_name"],
                        graph=dowhy_data["gml_graph"])
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    linear_estimate = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=0,
                                            treatment_value=1,
                                            method_params={'need_conditional_estimates': False},
                                            evaluate_effect_strength=False)
    
    return linear_estimate.value
    
def causal_fairness_indirect_effect(target=None, pred=None, x=None, prot_attr=None, dowhy_data=None, name=None):
    if "Indirect" not in name:
        return 0
    
    dowhy_data['df'] = dowhy_data['df'].iloc[:pred.shape[0]]
    dowhy_data['df'][dowhy_data['df'].columns[:-1]] = x
    dowhy_data['df']['y'] = pred[:,1]
    
    model = CausalModel(data=dowhy_data["df"],
                        treatment=dowhy_data["instrument_names"], outcome=dowhy_data["outcome_name"],
                        graph=dowhy_data["gml_graph"])
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    linear_estimate = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=0,
                                            treatment_value=1,
                                            method_params={'need_conditional_estimates': False},
                                            evaluate_effect_strength=False)
    
    return linear_estimate.value
     



     
