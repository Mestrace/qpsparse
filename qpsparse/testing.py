from __future__ import absolute_import

from qpsparse.parse import columns2df

def validate_columns2df(ast):
    """
    For each of the values in the COLUMNS, validate the values in the
    DataFrame; raises assertion error if inequality happens

    Parameters
    ----------
    ast : abstract syntax tree
        The abstract syntax tree of the MPS / QPS file

    Returns
    -------
    None
    """
    df = columns2df(ast)
    for col_name, row_name, value in ast['COLUMNS']:
        assert df.loc[row_name, col_name] == value
