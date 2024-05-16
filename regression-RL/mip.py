from gurobipy import *
import numpy as np


def generateCSP(L, n, A):
    m = Model('CG shapelet - Integer MP')
    #m.setParam('OutputFlag', 0)
    z = m.addVars(L, vtype=GRB.BINARY, name="z")
    m.addConstrs((quicksum(A[i, j] * z[j] for j in range(L)) == 1 for i in range(n)), name='con1')
    #m.update()
    return m, z


def generateProblem(L, n, A, l, loss, freq):
    m = Model('CG shapelet - Integer MP')
    #m.setParam('OutputFlag', 0)
    z = m.addVars(L, vtype=GRB.BINARY, name="z")
    m.addConstrs((quicksum(A[i, j] * z[j] for j in range(L)) == 1 for i in range(n)), name='con1')
    m.addConstr(quicksum(loss[j] * z[j] for j in range(L)) <= 0, name="ub_loss")
    m.setObjective(quicksum(freq[j] * z[j] for j in range(L)), sense=GRB.MAXIMIZE)

    m.addConstr(sum(z.select('*')) <= l, name='card')
    m.update()
    return m


def generateProblemSoft(L, n, A, l, loss, freq, lambd):
    m = Model('CG shapelet - Integer MP')
    m.setParam('OutputFlag', 0)
    z = m.addVars(L, vtype=GRB.BINARY, name="z")
    m.addConstrs((quicksum(A[i, j] * z[j] for j in range(L)) == 1 for i in range(n)), name='con1')
    m.setObjective(quicksum(freq[j] * z[j] for j in range(L)) - lambd*quicksum(loss[j] * z[j] for j in range(L)),
                               sense=GRB.MAXIMIZE)

    m.addConstr(sum(z.select('*')) <= l, name='card')
    m.update()
    return m, z