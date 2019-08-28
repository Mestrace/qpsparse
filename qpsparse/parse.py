from __future__ import absolute_import

import numpy as np
import pandas as pd
import tatsu
from scipy.sparse import coo_matrix

from collections import defaultdict

# The EBNF grammar for the MPS / QPS files
# noinspection PyPep8,SpellCheckingInspection
QPS_GRAMMAR = """
@@grammar::qps

start = identifiers;

identifiers = iname irow icolumns irhs [ iranges ] [ ibounds ] [iquadobj] 
iendata$;

iname = 'NAME' name;

name = /[\w\-\.]+/;

irow = 
    'ROWS' {row_record}+;

row_record = row_sense name;

row_sense = 
        | 'N' 
        | 'G' 
        | 'L' 
        | 'E'
        ;

icolumns = 'COLUMNS' {record}*;

irhs = 'RHS' {rhs_record}*;

rhs_record = name name number [name number];

iranges = 'RANGES' {record}*;

ibounds = 'BOUNDS' {bound_record}*;

bound_record = 
    | value_bound_record 
    | novalue_bound_record
    ;

value_bound_record = value_bound_type name name number;

novalue_bound_record = novalue_bound_type name name;

iquadobj = 'QUADOBJ' {record}*;

iendata = 'ENDATA';

value_bound_type = 
            | 'LO'
            | 'UP'
            | 'FX' 
            ;

novalue_bound_type = 
            | 'FR'
            | 'MI'
            | 'PL'
            ;

record = name name number [ name number ];

number = 
        | exponent 
        | decimal;

decimal = /-?(\d+\.?\d*|\.\d+)/;


exponent = /[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)/;
"""


def qps2ast(qps: str):
    """
    Parse qps string to its abstract syntax tree (ast) based on EBNF syntax for
    QPS

    Parameters
    ----------
    qps : str
        qps file in string

    Returns
    -------
    ast : dict
        The beautified abstract syntax tree of the input qps string indexed by
        its labels
    """
    raw_ast = tatsu.parse(QPS_GRAMMAR, qps)
    ast = decorate_ast(raw_ast)
    return ast


def decorate_ast(raw_ast):
    """
    Decorate the raw abstract syntax tree to reasonable structures; original
    MPS design is sick...

    Parameters
    ----------
    raw_ast : raw abstract syntax tree
        The raw abstract syntax tree of the MPS / QPS file coming from the
        EBNF parser.

    Returns
    -------
    Decorated version of the raw ast
    """
    def decorate_columns(columns):
        res = []
        for line in columns:
            res.append([line[0], line[1], float(line[2])])
            if len(line) == 5:
                res.append([line[0], line[3], float(line[4])])
        return res

    def decorate_rhs(rhs):
        res = []
        for line in rhs:
            res.append([line[0], line[1], float(line[2])])
            if len(line) == 5:
                res.append([line[0], line[3], float(line[4])])
        return res

    def decorate_ranges(ranges):
        res = []
        for line in ranges:
            raise ValueError("Invalid line format:\n\t%s" % str(line))
        return res

    def decorate_bounds(bounds):
        res = []
        for line in bounds:
            if len(line) == 3:
                res.append(line)
            elif len(line) == 4:
                res.append([line[0], line[1], line[2], float(line[3])])
            else:
                raise ValueError("Invalid line format:\n\t%s" % str(line))
        return res

    def decorate_quadobj(quadobj):
        res = []
        for line in quadobj:
            res.append([line[0], line[1], float(line[2])])
            if len(line) == 5:
                res.append([line[0], line[3], float(line[4])])
        return res

    decorate_kw2action = {
        "NAME": lambda name: name,
        "ROWS": lambda rows: rows,
        "COLUMNS": decorate_columns,
        "RHS": decorate_rhs,
        "RANGES": decorate_ranges,
        "BOUNDS": decorate_bounds,
        "QUADOBJ": decorate_quadobj,
    }

    ast = {}
    i = 0
    while i < len(raw_ast) - 1:  # ignore ENDATA
        if not isinstance(raw_ast[i], list):
            key, value = raw_ast[i], raw_ast[i + 1]
            i += 2
        else:
            key, value = raw_ast[i]
            i += 1

        ast[key] = decorate_kw2action[key](value)
    return ast


def columns2df(ast):
    """
    Convert COLUMNS to DataFrame for easier access and processing.

    Parameters
    ----------
    ast :  abstract syntax tree
        The abstract syntax

    Returns
    -------
    a pandas.DataFrame object represents the data in the COLUMNS in the form
    of matrix.
    """
    rows = [name for _, name in ast['ROWS']]
    cols = []

    coo_data = []
    coo_row_idx = []
    coo_col_idx = []

    for col_name, row_name, value in ast['COLUMNS']:
        coo_data.append(value)
        if col_name not in cols:
            cols.append(col_name)
        coo_row_idx.append(rows.index(row_name))
        coo_col_idx.append(cols.index(col_name))

    coo_mat = coo_matrix(
        (coo_data, (coo_row_idx, coo_col_idx)), shape=(len(rows), len(cols)))
    return pd.DataFrame.sparse.from_spmatrix(coo_mat, index=rows, columns=cols)


def bounds2np(ast, column_data_frame):
    """
    Convert BOUNDS to variable bounds expressed as inequality / equality
    constraints in the matrix form.

    Parameters
    ----------
    ast : abstract syntax tree
        The abstract syntax tree of the MPS/QPS objects
    column_data_frame : pandas.DataFrame
        the column DataFrame constructed from ROWS and COLUMNS

    Returns
    -------
    G : (M, N) array
        The left-hand side coefficients of the inequality constraints
    h : (M, ) array
        The right-hand side constants of the inequality constraints
    A : (K, N) array
        The left-hand side coefficients of the equality constraints
    c : (K, ) array
        The right-hand side constants of the equality constraints
    """
    # column_name_list is the set of the column names (optimization object)
    column_name_list = column_data_frame.columns.to_list()
    column_name_has_bound = [0 for _ in range(len(column_name_list))]

    # Gx <= h
    G = []
    h = []
    # Ax == c
    A = []
    c = []

    # noinspection PyShadowingNames,PyPep8Naming
    def convert_LO(idx, value):
        # -x <= -b
        arr = [0 for _ in range(len(column_name_list))]
        arr[idx] = -1
        G.append(arr)
        h.append(-value)

    # noinspection PyShadowingNames,PyPep8Naming
    def convert_UP(idx, value):
        # x <= b
        arr = [0 for _ in range(len(column_name_list))]
        arr[idx] = 1
        G.append(arr)
        h.append(value)

    # noinspection PyShadowingNames,PyUnusedLocal,PyPep8Naming
    def convert_FX(idx, value):
        # x = b
        arr = [0 for _ in range(len(column_name_list))]
        arr[idx] = 1
        A.append(arr)
        c.append(value)

    # noinspection PyUnusedLocal,PyShadowingNames,PyPep8Naming
    def convert_FR(idx, value=None):
        # do nothing
        pass

    # noinspection PyUnusedLocal,PyShadowingNames,PyPep8Naming
    def convert_MI(idx, value=None):
        convert_UP(idx, 0)

    # noinspection PyUnusedLocal,PyShadowingNames,PyPep8Naming
    def convert_PL(idx, value=None):
        convert_LO(idx, 0)

    bound_indicator2action = {
        # lower bound, b <= x <= inf ==> -x <= -b
        "LO": convert_LO,
        # upper bound, 0 <= x <= b   ==> x <= b && -x <= 0
        "UP": convert_UP,
        # fixed variable, x = b
        "FX": convert_FX,
        # free variable, -inf < x < +inf ==> nothing
        "FR": convert_FR,
        # lower bound -inf, -inf < x <= 0 ==>  x <= 0
        "MI": convert_MI,
        # upper bound +inf, 0 <= x < +inf  ==> -x <= 0
        "PL": convert_PL,
        # if not specified, PL
    }

    # iterate over all the values in BOUNDS and add to BOUNDS sections
    for ind, bound_name, column_name, *value in ast['BOUNDS']:
        idx = column_name_list.index(column_name)
        column_name_has_bound[idx] = 1
        bound_indicator2action[ind](idx, *value)

    # values that does not have a explicit bound gets PL
    for idx, has_bound in enumerate(column_name_has_bound):
        if not has_bound:
            convert_PL(idx)

    return (np.asarray(G, dtype=float).reshape((-1, column_data_frame.shape[1])),
            np.asarray(h, dtype=float).reshape(-1),
            np.asarray(A, dtype=float).reshape((-1, column_data_frame.shape[1])),
            np.asarray(c, dtype=float).reshape(-1))


def rhs2dict(ast):
    """
    Converts RHS tag to dictionary indexed by the row name; during the
    conversion the name of the RHS condition is discarded since it does not
    convey useful information.

    Parameters
    ----------
    ast : the abstract syntax tree

    Returns
    -------
    A
    """
    # discards the RHS condition name:
    res = defaultdict(lambda : 0.0)
    for _, row_name, value in ast['RHS']:
        res[row_name] = value
    return res


def rows2np(ast, column_dataframe):
    """
    Convert ROWS and COLUMNS to inequality / equality constraints in the
    matrix form

    Parameters
    ----------
    ast : abstract syntax tree
        The abstract syntax tree of the MPS/QPS objects
    column_dataframe :
        the column dataframe constructed from ROWS and COLUMNS

    Returns
    -------
    G : (M, N) array
        The left-hand side coefficients of the inequality constraints
    h : (M, ) array
        The right-hand side constants of the inequality constraints
    A : (K, N) array
        The left-hand side coefficients of the equality constraints
    c : (K, ) array
        The right-hand side constants of the equality constraints
    """
    rhs_rows2val = rhs2dict(ast)

    G = []
    h = []
    A = []
    c = []

    # noinspection PyShadowingNames,PyPep8Naming
    def convert_E(row_name):
        lhs = column_dataframe.loc[row_name]
        rhs = rhs_rows2val[row_name]
        A.append(lhs)
        c.append(rhs)

    # noinspection PyShadowingNames,PyPep8Naming
    def convert_L(row_name):
        lhs = column_dataframe.loc[row_name]
        rhs = rhs_rows2val[row_name]
        G.append(lhs)
        h.append(rhs)

    # noinspection PyShadowingNames,PyPep8Naming
    def convert_G(row_name):
        lhs = column_dataframe.loc[row_name]
        rhs = rhs_rows2val[row_name]
        G.append(-lhs)
        h.append(-rhs)

    constraints_sense2action = {
        'N': lambda *x: None,
        'E': convert_E,
        "L": convert_L,
        "G": convert_G,
    }

    for sense, row_name in ast['ROWS']:
        constraints_sense2action[sense](row_name)

    return (np.asarray(G, dtype=float).reshape((-1, column_dataframe.shape[1])),
            np.asarray(h, dtype=float).reshape(-1),
            np.asarray(A, dtype=float).reshape((-1, column_dataframe.shape[1])),
            np.asarray(c, dtype=float).reshape(-1))

def quadobj2np(ast, column_dataframe):
    """
    Convert QUADOBJ to numpy array in the form of matrix

    Parameters
    ----------
    ast : abstract syntax tree
        The abstract syntax tree of the MPS/QPS objects
    column_dataframe :
        the column dataframe constructed from ROWS and COLUMNS

    Returns
    -------
    quadobj : (N, N) array
        the quadratic objective coefficient of the objective function
    """

    column_name_list = column_dataframe.columns.to_list()

    quadobj = np.zeros((len(column_name_list), len(column_name_list)), dtype=float)

    for x_name, y_name, value in ast['QUADOBJ']:
        quadobj[
            column_name_list.index(y_name),
            column_name_list.index(x_name)
        ] = value
    return quadobj

def ast2np(ast):
    """
    Convert MPS/QPS ast to numpy arrays

    Parameters
    ----------
    ast : abstract syntax tree
        The abstract syntax tree of the MPS/QPS objects

    Returns
    -------
    Q : (N, N) array
        The quadratic coefficient of the objective function; if the input is
        MPS, the quadratic coefficient will be all zeros.
    b : (N, ) array
        The linear coefficient of the objective function
    G : (M, N) array
        The left-hand side coefficients of the inequality constraints
    h : (M, ) array
        The right-hand side constants of the inequality constraints
    A : (K, N) array
        The left-hand side coefficients of the equality constraints
    c : (K, ) array
        The right-hand side constants of the equality constraints
    """
    # Get the first occurrence of N in ROWS
    linear_objective_function_name = None
    for sense, name in ast['ROWS']:
        if sense == 'N':
            linear_objective_function_name = name
            break
    if linear_objective_function_name is None:
        raise ValueError("No objective function found.")

    # convert COLUMNS to pandas DF for easier access
    column_dataframe = columns2df(ast)

    # Get the linear part of objective function
    b = column_dataframe.loc[linear_objective_function_name].to_numpy()
    column_dataframe.drop([linear_objective_function_name])

    # Get the quadratic part of objective function
    if 'QUADOBJ' in ast:
        Q = quadobj2np(ast, column_dataframe)
        Q = np.tril(Q, -1).T + Q
    else:
        Q = np.zeros((b.shape[0], b.shape[0]), dtype=float)

    # Deal with linear constraint
    Gs = []
    hs = []
    As = []
    cs = []

    # Deal with rows
    G, h, A, c = rows2np(ast, column_dataframe)
    Gs.append(G)
    hs.append(h)
    As.append(A)
    cs.append(c)

    if 'BOUNDS' in ast:
        G, h, A, c = bounds2np(ast, column_dataframe)
        Gs.append(G)
        hs.append(h)
        As.append(A)
        cs.append(c)

    if 'RANGES' in ast:
        pass
        # G, h, A, c = bounds2np(ast, column_dataframe)
        # Gs.append(G)
        # hs.append(h)
        # As.append(A)
        # cs.append(c)

    G = np.vstack(Gs)
    h = np.concatenate(hs)
    A = np.vstack(As)
    c = np.concatenate(cs)

    return Q, b, G, h, A, c


def qps2np(qps: str):
    """
    Wrapper function for qpsparse.qps2ast and qpsparse.ast2np; converts MPS /
    QPS string to numpy objects

    Parameters
    ----------
    qps : str
        The QPS string

    Returns
    -------
    Q : (N, N) array
        The quadratic coefficient of the objective function; if the input is
        MPS, the quadratic coefficient will be all zeros.
    b : (N, ) array
        The linear coefficient of the objective function
    G : (M, N) array
        The left-hand side coefficients of the inequality constraints
    h : (M, ) array
        The right-hand side constants of the inequality constraints
    A : (K, N) array
        The left-hand side coefficients of the equality constraints
    c : (K, ) array
        The right-hand side constants of the equality constraints

    See Also
    --------
    qpsparse.qps2ast : MPS / QPS string to decorated abstract syntax tree
    qpsparse.ast2np : converts abstract syntax tree to numpy objects
    """
    ast = qps2ast(qps)
    return ast2np(ast)

def qpsfile2np(fn : str):
    """
    Wrapper function for qpsparse.qps2np; converts MPS / QPS file to numpy
    objects

    Parameters
    ----------
    fn : str
        The relative / absolute path to the MPS / QPS file

    Returns
    -------
    Q : (N, N) array
        The quadratic coefficient of the objective function; if the input is
        MPS, the quadratic coefficient will be all zeros.
    b : (N, ) array
        The linear coefficient of the objective function
    G : (M, N) array
        The left-hand side coefficients of the inequality constraints
    h : (M, ) array
        The right-hand side constants of the inequality constraints
    A : (K, N) array
        The left-hand side coefficients of the equality constraints
    c : (K, ) array
        The right-hand side constants of the equality constraints

    See Also
    --------
    qpsparse.qps2ast : MPS / QPS string to decorated abstract syntax tree
    qpsparse.ast2np : converts abstract syntax tree to numpy objects
    qpsparse.qps2np : converts MPS / QPS string to numpy objects

    """
    with open(fn, 'r') as problem_file:
        return qps2np(problem_file.read())
