import torch
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Union


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    # 1. Find p(x).
    # p(x) is the prob an outcome will happen (e.g. we get a label of x class)
    # p(x) = num times x appears in labels / length of labels
    total_length = len(labels)
    label_probs = Counter(labels) # returns a dictionary
    probs = torch.tensor([label_prob/total_length for label_prob in label_probs.values()])
    
    # 2. Calculate entropy and return
    return -(probs * torch.log2(probs)).sum(dim=-1).item()

def calculate_information_gain(
    examples: List[Dict[str, Any]],
    attr: str,
    target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """
    # Information gain = entropy of dataset - sum of ((|D where A has value v|/|D|)*entropy of D where A has value v)

    # D where A has value v == Dv
    # A is an attribute, target_attr is what we're trying to predict
    # v is a value A can take
    # in the example, if A = 'Outlook', then v could be 'Overcast', 'Sunny', or 'Rain'

    # 1. Define Dv
    Dv = defaultdict(list)
    for ex in examples: # for each row in the dataset
        # append the attribute value as key, and then the entire row as the value
        Dv[ex[attr]].append(ex)
        # This creates a dictionary of all the possible values of A mapped to the entire row (aka, the subset of the dataset where attribute A takes on value v)
    
    # 2. Calculate info gain

    # Sum the entropy of each subset of Dv using weighted_average
    conditional_entropy = 0
    # conditional_entropy represents the amount of uncertainty remaining in a variable (target_attr) after the value of another variable (attr) is known
    for data_subset in Dv.values():
        # calculate entropy fails with a list of dicts
        # all we need are the labels of target_attr within the data_subset
        labels = [row[target_attr] for row in data_subset]
        entropy = calculate_entropy(labels)
        weighted_avg_ratio = len(data_subset)/len(examples)
        conditional_entropy += (weighted_avg_ratio * entropy)
    
    # again, calc entropy fails on a list of dicts, so just get a list of the labels that target_attr takes on
    dataset_labels = [row[target_attr] for row in examples]
    info_gain = calculate_entropy(dataset_labels) - conditional_entropy

    # 3. Return info gain
    return info_gain

def majority_class(
    examples: List[Dict[str, Any]],
    target_attr: str
) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    In case of a tie, return the class that comes first alphabetically.
    """
    # 1. Get target values
    # gets the value of target_attr for each row in the dataset
    target_vals = [ex[target_attr] for ex in examples]

    # 2. Find most frequent value
    # maps each class to its count in a dict
    counter = Counter(target_vals)

    # gets the max frequency of all the values for all the labels
    max_freq = max(counter.values())

    # creates a dictionary of all items with the max frequency
    max_freq_items = [label for label, freq in counter.items() if freq == max_freq]

    # 3. Alphabetical sort
    alphabetical_order_max_freq_items = sorted(max_freq_items)

    # 4. Return the label that has the most frequent value for target_attr and is highest in the alphabet of those with the most frequent value
    return alphabetical_order_max_freq_items[0]

def learn_decision_tree(
    examples: List[Dict[str, Any]],
    attributes: List[str],
    target_attr: str
) -> Union[Dict[str, Any], Any]:
    """
    Learn a decision tree using the ID3 algorithm.
    Returns either a nested dict representing the tree or a class label at the leaves.
    """
    # 4. Base cases. If all examples have the same value for target_attr, return that. If no attributes remain, return the majority class.
    # If examples have same value for target_attr, return that value
    target_attr_values = []
    for row in examples:
        target_attr_values.append(row[target_attr])
    target_attr_values = set(target_attr_values)
    if len(target_attr_values) == 1: return list(target_attr_values)[0]

    # if no attributes remain, return the majority class
    if len(attributes) == 0:
        return majority_class(examples, target_attr)
    
    # 1. Find attribute with the highest information gain
    attr_highest_info_gain = ""
    highest_info_gain = -float('inf')
    for attr in attributes:
        attr_info_gain = calculate_information_gain(examples, attr, target_attr)
        if attr_info_gain > highest_info_gain:
            highest_info_gain = attr_info_gain
            attr_highest_info_gain = attr

    # 2. Divide dataset on the values of the selected attribute)
    subsets = defaultdict(list)
    for row in examples:
        subsets[row[attr_highest_info_gain]].append(row)
        # the subsets dict stores each value and its row(s) for the attribute with best info gain
        # key is the attribute value
        # value is the list of examples corresponding to that attribute value

    # 3. Recursion: repeat until perfectly classified or no remaining attributes can be used for split
    remaining_attributes = attributes[:attributes.index(attr_highest_info_gain)] + attributes[attributes.index(attr_highest_info_gain) + 1:]

    # Recurse on each subset of the subsets dict to get the tree
    tree = {}
    for attr_val, subset in subsets.items():
        tree[attr_val] = learn_decision_tree(subset, remaining_attributes, target_attr)
    
    # return the tree, wrapped in a dict with key attr_highest_info_gain as the first split made
    return {attr_highest_info_gain: tree}
