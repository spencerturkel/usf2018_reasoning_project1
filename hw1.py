import string
from enum import Enum, auto
from string import ascii_lowercase, digits
from typing import Union


# Formula Grammar
###########################################################
# F = V | (L)
# V = aV' | bV' | ... | zV' | 0V' | 1V' | ... | 9V'
# V' = aV' | bV' | ... | zV' | 0V' | 1V' | ... | 9V' | ''
# L = O F F | 'NOT' F
# O = 'IF' | 'AND' | 'OR'
###########################################################

class Op(Enum):
    """
    Operations supported in formulas.
    """
    AND = auto()
    IF = auto()
    NOT = auto()
    OR = auto()


def parse(formula: str):
    """
    Parses a formula string into an AST tuple representation using Op.

    >>> parse('(AND (IF p q) (NOT r))')
    (<Op.AND: 1>, (<Op.IF: 2>, 'p', 'q'), (<Op.NOT: 3>, 'r'))
    >>> parse('q1')
    'q1'
    >>> parse('ABC') is None
    True
    >>> parse('((NOT r)') is None
    True
    >>> parse('(NOT r))') is None
    True
    >>> parse('((NOT r))') is None
    True
    >>> parse('abCdef') is None
    True
    >>> parse('(F q)') is None
    True
    >>> parse('') is None
    True
    """
    # Implemented as a recursive-descent parser.
    length = len(formula)
    index = 0  # Current view into formula

    def consume_whitespace():
        """Increases the index while the current token is whitespace."""
        nonlocal formula, index
        while index < length and formula[index] in string.whitespace:
            index += 1

    def parse_form():
        """Parses a formula."""
        nonlocal formula, index
        if index >= length:
            raise ValueError('Empty form')

        ch = formula[index]
        if ch == '(':
            index += 1
            consume_whitespace()
            result = parse_list()
            consume_whitespace()
            if index < length and formula[index] == ')':
                index += 1
                return result
            raise ValueError('Unclosed form')
        if ch in (ascii_lowercase + digits):
            word = []
            while index < length and formula[index] in (ascii_lowercase + digits):
                word.append(formula[index])
                index += 1
            return ''.join(word)
        raise ValueError('Could not parse form')

    def parse_list():
        """Parses the contents of a parenthesized s-exp list"""
        nonlocal formula, index
        for op in Op:
            if formula.startswith(op.name, index):
                index += len(op.name)
                consume_whitespace()
                first_arg = parse_form()
                if op == Op.NOT:
                    return op, first_arg
                consume_whitespace()
                second_arg = parse_form()
                return op, first_arg, second_arg
        raise ValueError('Could not parse list')

    consume_whitespace()
    try:
        final_result = parse_form()
        return final_result if index == length else None
    except ValueError:
        return None


# noinspection PyPep8Naming
def proveFormula(formula: str) -> Union[int, str]:
    parseResult = parse(formula)

    if parseResult is None:
        return 'E'

    # TODO
    return formula
