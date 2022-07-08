# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 14:35
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: pgmpy_test.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from pprint import pprint
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.estimators import ExhaustiveSearch

## Build bayesian networks
# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

# CPDs can also be defined using the state names of the variables. If the state names are not provided
# like in the previous example, pgmpy will automatically assign names as: 0, 1, 2, ....
cpd_d_sn = TabularCPD(variable='D', variable_card=2, values=[[0.6], [0.4]], state_names={'D': ['Easy', 'Hard']})
cpd_i_sn = TabularCPD(variable='I', variable_card=2, values=[[0.7], [0.3]], state_names={'I': ['Dumb', 'Intelligent']})
cpd_g_sn = TabularCPD(variable='G', variable_card=3,
                      values=[[0.3, 0.05, 0.9, 0.5],
                              [0.4, 0.25, 0.08, 0.3],
                              [0.3, 0.7, 0.02, 0.2]],
                      evidence=['I', 'D'],
                      evidence_card=[2, 2],
                      state_names={'G': ['A', 'B', 'C'],
                                   'I': ['Dumb', 'Intelligent'],
                                   'D': ['Easy', 'Hard']})
cpd_l_sn = TabularCPD(variable='L', variable_card=2,
                      values=[[0.1, 0.4, 0.99],
                              [0.9, 0.6, 0.01]],
                      evidence=['G'],
                      evidence_card=[3],
                      state_names={'L': ['Bad', 'Good'],
                                   'G': ['A', 'B', 'C']})
cpd_s_sn = TabularCPD(variable='S', variable_card=2,
                      values=[[0.95, 0.2],
                              [0.05, 0.8]],
                      evidence=['I'],
                      evidence_card=[2],
                      state_names={'S': ['Bad', 'Good'],
                                   'I': ['Dumb', 'Intelligent']})

# These defined CPDs can be added to the model. Since, the model already has CPDs associated to variables, it will
# show warning that pmgpy is now replacing those CPDs with the new ones.
model.add_cpds(cpd_d_sn, cpd_i_sn, cpd_g_sn, cpd_l_sn, cpd_s_sn)
model.check_model()

## Get states of network
# We can now call some methods on the BayesianModel object.
pprint(model.get_cpds())
# Printing a CPD with it's state names defined.
print(model.get_cpds('G'))
print(model.get_cardinality('G'))

## Independencies in Bayesian Networks
# Getting all the local independencies in the network.
print(model.local_independencies(['D', 'I', 'S', 'G', 'L']))
# Active trail: For any two variables A and B in a network if any change in A influences the values of B then we say
#               that there is an active trail between A and B.
# In pgmpy active_trail_nodes gives a set of nodes which are affected (i.e. correlated) by any
# change in the node passed in the argument.
model.active_trail_nodes('D')
model.active_trail_nodes('D', observed='G')

## Variable Elimination
infer = VariableElimination(model)
g_dist = infer.query(['G'])
print(g_dist)
print(infer.query(['G'], evidence={'D': 'Easy', 'I': 'Intelligent'}))

## Predicting values from new data points
print(infer.map_query(['G']))

## Scoring functions
# create random data sample with 3 variables, where Z is dependent on X, Y:
data1 = pd.DataFrame(np.random.randint(0, 4, size=(5000, 2)), columns=list('XY'))
data1['Z'] = data1['X'] + data1['Y']
bdeu1 = BDeuScore(data1, equivalent_sample_size=5)

data2 = pd.DataFrame(np.random.randint(0, 4, size=(5000, 2)), columns=list('XZ'))
data2['Y'] = data2['X'] + data2['Z']
bdeu2 = BDeuScore(data2, equivalent_sample_size=5)

model = BayesianModel([('X', 'Y'), ('Z', 'Y')])  # X -> Z <- Y

print(bdeu1.score(model))
print(bdeu2.score(model))

## Search strategies
es = ExhaustiveSearch(data2, scoring_method=bdeu2)
best_model = es.estimate()
print(best_model.edges())

print("\nAll DAGs by score:")
for score, dag in reversed(es.all_scores()):
    print(score, dag.edges())

from pgmpy.utils import get_example_model
model = get_example_model("asia")
phi = model.get_cpds("either").to_factor()
phi.get_value(lung="no", tub="no", either="yes")
