from gurobipy import *
import numpy as np


def generateCSP(L, n, A):
    m = Model('Integer MP')
    #m.setParam('OutputFlag', 0)
    z = m.addVars(L, vtype=GRB.BINARY, name="z")
    m.addConstrs((quicksum(A[i, j] * z[j] for j in range(L)) == 1 for i in range(n)), name='con1')
    #m.update()
    return m, z


def generateProblemSoft(L, n, A, l, loss, freq, lambd=0.5):
    m = Model('Integer MP')
    m.setParam('OutputFlag', 0)
    z = m.addVars(L, vtype=GRB.BINARY, name="z")
    m.addConstrs((quicksum(A[i, j] * z[j] for j in range(L)) == 1 for i in range(n)), name='con1')
    m.setObjective(quicksum(- (1 - lambd) * loss[j] * z[j] for j in range(L)) + lambd*quicksum(freq[j] * z[j] for j in range(L)),
                               sense=GRB.MAXIMIZE)

    m.addConstr(sum(z.select('*')) <= l, name='card')
    m.update()
    return m, z
