"""Gurobi set-partitioning formulation shared by all experiments."""

import warnings

import numpy as np
from scipy import sparse


class NoSolutionError(RuntimeError):
    """The optimization run did not produce a feasible incumbent."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def coverage_matrix(samples, n_samples):
    """Build sparse row-to-rule incidence without an n-by-L dense allocation."""
    rows = np.concatenate(samples)
    cols = np.repeat(np.arange(len(samples)), [len(sample) for sample in samples])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_samples, len(samples)))


def generateCSP(L, n, A, *, time_limit=None, threads=1, seed=0, verbose=False):
    """Build Az=1, z binary. Import Gurobi only when optimization is requested."""
    import gurobipy as gp

    A = sparse.csr_matrix(A)
    if A.shape != (n, L) or n < 1 or L < 1:
        raise ValueError("A must have shape (n, L) with positive dimensions.")
    if not np.isfinite(A.data).all() or np.any((A.data != 0) & (A.data != 1)):
        raise ValueError("A must be a binary coverage matrix.")
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be positive and finite.")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    model = gp.Model("TEDIP set partitioning")
    model.Params.OutputFlag = int(verbose)
    model.Params.Threads = threads
    model.Params.Seed = int(seed)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    z = model.addVars(L, vtype=gp.GRB.BINARY, name="z")
    for i in range(n):
        row = A.getrow(i)
        model.addConstr(
            gp.quicksum(value * z[int(j)] for j, value in zip(row.indices, row.data)) == 1,
            name=f"cover[{i}]",
        )
    return model, z


def generateProblemSoft(L, n, A, l, loss, freq, lambd=0.5, **solver_options):  # noqa: E741
    """Maximize lambda*stability - (1-lambda)*loss with at most l rules.

    Coverage constraints are hard equality constraints.
    Pass l=None to omit the cardinality bound.
    """
    import gurobipy as gp

    if not np.isfinite(lambd) or not 0 <= lambd <= 1:
        raise ValueError("lambd must lie between 0 and 1.")
    if l is not None and (not isinstance(l, (int, np.integer)) or l < 1):
        raise ValueError("leaf_nodes must be a positive integer or None.")
    loss, freq = np.asarray(loss), np.asarray(freq)
    if loss.shape != (L,) or freq.shape != (L,) or not np.isfinite([loss, freq]).all():
        raise ValueError("loss and freq must be finite vectors of length L.")
    model, z = generateCSP(L, n, A, **solver_options)
    model.setObjective(
        gp.quicksum((lambd * freq[j] - (1 - lambd) * loss[j]) * z[j] for j in range(L)),
        gp.GRB.MAXIMIZE,
    )
    if l is not None:
        model.addConstr(z.sum() <= l, name="card")
    model.update()
    return model, z


def selected_rules(model, z, *, require_optimal=False):
    """Check status before reading solution values, including time-limit exits."""
    import gurobipy as gp

    if model.SolCount == 0:
        raise NoSolutionError(
            f"Gurobi returned status {model.Status} with no feasible solution. "
            "Increase the rule budget/time limit, or reduce Nmin.",
            status=model.Status,
        )
    if model.Status != gp.GRB.OPTIMAL:
        if require_optimal:
            raise NoSolutionError("Exact rule-count bounds require an optimal solve.")
        warnings.warn(
            f"Using a feasible, nonoptimal solution (status {model.Status}, "
            f"gap {model.MIPGap:.3g}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return [j for j in range(len(z)) if z[j].X > 0.5]
