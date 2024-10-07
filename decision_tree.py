"""Reference: Data Science from scratch by Joel Grus"""

from typing import Dict, List
import math
import collections
from dataclasses import dataclass

@dataclass
class Input:
    details: Dict[str, str]
    label: bool

def entropy(class_probabilities: List[float]) -> float:
    """
    Entropy: H = -p1 * log(p1) - p2 * log(p2) - ... - pn * log(pn)
    
    Args:
        class_probabilities: list of probabilities for different classes in the dataset
                             Eg: [0.1, 0.3, 0.2, ..]
    
    """
    H = sum([-p * math.log(p, 2)
             for p in class_probabilities
            if p])
    
    return H

def class_probabilities(labels: List[bool]) -> List[float]:
    """
    """
    total_count = len(labels)
    return [count/total_count
            for count in Counter(labels).values()]

def data_entropy(inputs: List[[Dict[str, str], bool]]):
    labels = [label for _, label in inputs]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_by(inputs: List[Input], attribute) -> Dict[str, List]:
    partitions = collections.defaultdict(list)
    for input in inputs:
        attribute_value = input[0][attribute]
        partitions[attribute_value].append(input)
    return partitions
    
def partition_entropy(subsets):
    """ H = q1*H(S1) + q2*H(S2) + ..
    """
    total_count = sum([len(subset) for subset in subsets])
    return sum([(len(s)/total_count) * data_entropy(s)
                for s in subsets])
    
def partition_entropy_by(inputs, attribute):
    # get partitions for each possible value of attributes
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())
    
def train_id3(inputs, split_candidates = None):
    "Trains decision tree using ID3 algorithm"
    if not split_candidates:
        split_candidates = inputs[0][0].keys()
        
    # Find the best attribute to split on
    best_attribute, best_partition = None, []
    min_entropy = float("inf")
    for candidate in split_candidates:
        # get total entropy for each candidate
        entropy, partitions = partition_entropy_by(inputs, candidate)
        
        print(entropy)
        print(partitions)
        
        # if entropy < min_entropy:
        #     min_entropy = entropy
        #     best_attribute = candidate
        #     best_partition = partitions
    

def classify(model, input: Dict[str, str]) -> bool:
    ...
    


if __name__ == "__main__":
    
    inputs = [
    ({'level': 'Senior', 'lang': 'Java', 'tweets':'no', 'phd': 'no'}, False),
    ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
    ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level':'Junior', 'lang':'R', 'tweets': 'yes', 'phd': 'yes'}, False),
    ({'level':'Mid', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, True),
    ({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, False),
    ({'level': 'Senior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level':'Junior', 'lang': 'Python', 'tweets':'yes', 'phd': 'no'}, True), 
    ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
    ({'level':'Mid', 'lang': 'Python', 'tweets':'no', 'phd': 'yes'}, True), 
    ({'level':'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True), 
    ({'level':'Junior', 'lang': 'Python', 'tweets':'no', 'phd': 'yes'}, False)
    ]
    
    model = train_id3(inputs)
    
    # Test model on sample data points
    sample1 = {
        'level': 'Junior',
        'lang' : 'Java',
        'tweets': 'yes',
        'phd': 'no'
    }
    
    sample2 = {
        'level': 'Junior',
        'lang' : 'Java',
        'tweets': 'yes',
        'phd': 'yes'
    }
    
    label = classify(model, sample1)
    assert label == True
    
    label = classify(model, sample2)
    assert label == False