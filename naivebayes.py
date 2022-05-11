from collections import defaultdict
import itertools
import pandas as pd


class JointDistribution:
    def __init__(self, variables, event_counts, smoother):
        self.variables = variables
        self.event_probs = smoother(event_counts)

    def prob(self, event):
        return self.event_probs[tuple(event)]


class CPT:
    def __init__(self, joint_dist, parent_dist):
        self.joint_dist = joint_dist
        self.parent_dist = parent_dist

    def prob(self, child_event, parent_event):
        numerator = self.joint_dist.prob(tuple([child_event] + parent_event))
        denominator = self.parent_dist.prob(tuple(parent_event))
        return numerator / denominator


class NaiveBayes:
    def __init__(self, signature, parent_name, parent_prior, cpts):
        self.signature = signature
        self.parent_name = parent_name
        self.parent_prior = parent_prior
        self.cpts = cpts

    def posterior(self, child_observations):
        joints = dict()
        for parent_value in self.signature[self.parent_name]:
            joints[parent_value] = self.parent_prior.prob(parent_value, [])
            for child, child_value in child_observations.items():
                joints[parent_value] *= self.cpts[child].prob(child_value, [parent_value])
        normalizer = sum([joints[p] for p in self.signature[self.parent_name]])
        return {parent_value: joints[parent_value] / normalizer
                for parent_value in self.signature[self.parent_name]}


def add_k_smoothing(event_counts, k):
    augmented = {event: event_counts[event] + k for event in event_counts}
    normalizer = sum([count for _, count in augmented.items()])
    return {event: augmented[event]/normalizer for event in augmented}


def extract_signature(data):
    signature = defaultdict(set)
    for _, row in data.iterrows():
        for column_heading in data:
            signature[column_heading].add(row[column_heading])
    return {k: tuple(sorted(signature[k])) for k in signature}


def train_cpt(data, child, parents):
    signature = extract_signature(data)
    domains = [signature[child]] + [signature[parent] for parent in parents]
    joint_events = [event for event in itertools.product(*domains)]
    parent_events = [event for event in itertools.product(*domains[1:])]
    joint_event_counts = {event: 0 for event in joint_events}
    parent_event_counts = {event: 0 for event in parent_events}
    for _, row in data.iterrows():
        instance = tuple([row[child]] + [row[parent] for parent in parents])
        joint_event_counts[instance] += 1
        parent_event_counts[instance[1:]] += 1
    joint_dist = JointDistribution([child] + parents, joint_event_counts,
                                   lambda c: add_k_smoothing(c, k=1))
    parent_dist = JointDistribution(parents, parent_event_counts,
                                    lambda c: add_k_smoothing(c, k=1))
    return CPT(joint_dist, parent_dist)


def train_naive_bayes(data, parent):
    signature = extract_signature(data)
    children = [name for name in signature if name != parent]
    parent_prior = train_cpt(data, parent, [])
    cpts = {child: train_cpt(data, child, [parent]) for child in children}
    return NaiveBayes(signature, parent, parent_prior, cpts)
