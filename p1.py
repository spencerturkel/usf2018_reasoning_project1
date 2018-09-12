import string
from enum import Enum
from itertools import combinations
from string import ascii_lowercase, digits


################################################################
# Formula Grammar
# S-exp = ws freevar ws | ws list ws
# ws = <empty> | <space> | <tab> | <newline> | <carriage return>
# freevar = [a-z0-9]+
# list = ( ws var-op ws S-exp ws S-exp-list ws )
# | ( ws NOT ws S-exp ws )
# | ( ws IF ws S-exp ws S-exp ws )
# var-op = AND | OR
# S-exp-list = S-exp | S-exp ws S-exp-list
################################################################


class Op(Enum):
    """
    Operations supported in formulas.
    """
    AND = 1
    IF = 2
    NOT = 3
    OR = 4


def parse(formula: str):
    """
    Parses a formula string into an AST tuple representation using Op.

    >>> parse('(AND (IF p q) (NOT r))')
    (<Op.AND: 1>, (<Op.IF: 2>, 'p', 'q'), (<Op.NOT: 3>, 'r'))
    >>> parse('(OR p (NOT q))')
    (<Op.OR: 4>, 'p', (<Op.NOT: 3>, 'q'))
    >>> parse('(AND (IF p q) (NOT r) (OR p q r))') # AND/OR take 2+ args
    (<Op.AND: 1>, (<Op.IF: 2>, 'p', 'q'), (<Op.NOT: 3>, 'r'), (<Op.OR: 4>, 'p', 'q', 'r'))
    >>> parse('q1')
    'q1'
    >>> parse('ABC') is None
    True
    >>> parse('(IF p q r)') is None # IF always takes 2 args
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
        if formula.startswith(Op.NOT.name, index):
            index += 3  # len('NOT')
            consume_whitespace()
            first_arg = parse_form()
            return Op.NOT, first_arg
        if formula.startswith(Op.IF.name, index):
            index += 2  # len(Op.IF)
            consume_whitespace()
            first_arg = parse_form()
            consume_whitespace()
            second_arg = parse_form()
            return Op.IF, first_arg, second_arg
        for op in [Op.AND, Op.OR]:
            if formula.startswith(op.name, index):
                index += len(op.name)
                consume_whitespace()
                first_arg = parse_form()
                call = [op, first_arg]
                consume_whitespace()
                while formula[index] != ')':
                    call.append(parse_form())
                    consume_whitespace()
                return tuple(call)
        raise ValueError('Could not parse list')

    consume_whitespace()
    try:
        final_result = parse_form()
        return final_result if index == length else None
    except ValueError:
        return None


def collect_free_vars(ast):
    """
    Gets the set of free variables in a formula.
    :param ast: Result from parse()

    >>> collect_free_vars('x')
    {'x'}
    >>> collect_free_vars((Op.NOT, 'x'))
    {'x'}
    >>> collect_free_vars((Op.OR, 'x', (Op.NOT, 'x')))
    {'x'}
    >>> collect_free_vars((Op.AND, 'x', (Op.NOT, 'y'))) == {'x', 'y'}
    True
    >>> collect_free_vars((Op.OR, 'x', (Op.OR, 'y', 'z'))) == {'x', 'y', 'z'}
    True
    """
    result = set()
    nodes_to_process = [ast]
    while nodes_to_process:
        node = nodes_to_process[0]
        nodes_to_process.pop(0)
        if isinstance(node, str):
            result.add(node)
            continue
        nodes_to_process.append(node[1])
        if Op.NOT == node[0]:
            continue
        nodes_to_process.append(node[2])
    return result


def evaluate(ast, true_vars):
    """
    Evaluates the AST, using specified variables as True and all others False.
    :param ast:
    :param true_vars:
    :return: Bool indicating whether the AST is satisfied
    >>> evaluate('a', {})
    False
    >>> evaluate('a', {'a', 'b'})
    True
    >>> evaluate((Op.IF, 'a', 'a'), {})
    True
    >>> evaluate((Op.IF, 'a', 'b'), {})
    True
    >>> evaluate((Op.IF, 'a', 'a'), {'a'})
    True
    >>> evaluate((Op.IF, 'a', 'b'), {'a'})
    False
    >>> evaluate((Op.NOT, 'a'), {})
    True
    >>> evaluate((Op.NOT, 'a'), {'a'})
    False
    >>> evaluate((Op.AND, 'a', 'b'), {})
    False
    >>> evaluate((Op.AND, 'a', 'b'), {'a'})
    False
    >>> evaluate((Op.AND, 'a', 'b'), {'a', 'b'})
    True
    >>> evaluate((Op.AND, 'a', 'b', 'c'), {'a', 'b'})
    False
    >>> evaluate((Op.AND, 'a', 'b', 'c'), {'a', 'b', 'c'})
    True
    >>> evaluate((Op.AND, 'a', 'b'), {'b'})
    False
    >>> evaluate((Op.OR, 'a', 'b'), {})
    False
    >>> evaluate((Op.OR, 'a', 'b'), {'a'})
    True
    >>> evaluate((Op.OR, 'a', 'b', 'c'), {'a'})
    True
    >>> evaluate((Op.OR, 'a', 'b', 'c'), {'c'})
    True
    >>> evaluate((Op.OR, 'a', 'b', 'c'), {})
    False
    >>> evaluate((Op.OR, 'a', 'b'), {'a', 'b'})
    True
    >>> evaluate((Op.OR, 'a', 'b'), {'b'})
    True
    """

    def go(node):  # closes over true_vars, since there are no changes to it
        if isinstance(node, str):
            return node in true_vars
        op = node[0]
        if op == Op.NOT:
            return not go(node[1])
        if op == Op.IF:
            return go(node[2]) if go(node[1]) else True
        if op == Op.AND:
            for sub_node in node[1:]:
                if not go(sub_node):
                    return False
            return True
        if op == Op.OR:
            for sub_node in node[1:]:
                if go(sub_node):
                    return True
            return False

    return go(ast)


def determine_satisfiability(ast):
    """
    Determines the satisfiability of a formula.
    :param ast: Result from parse()
    :return: True if tautology, False if unsatisfiable, otherwise number of satisfying variable instantiations

    >>> determine_satisfiability('x')
    1
    >>> determine_satisfiability((Op.NOT, 'x'))
    1
    >>> determine_satisfiability((Op.OR, 'x', (Op.NOT, 'x')))
    True
    >>> determine_satisfiability((Op.AND, 'x', (Op.NOT, 'x')))
    False
    >>> determine_satisfiability((Op.OR, 'x', 'y'))
    3
    >>> determine_satisfiability((Op.OR, 'x', (Op.OR, 'y', 'z')))
    7
    >>> determine_satisfiability((Op.NOT, (Op.OR, 'x', (Op.OR, 'y', 'z'))))
    1
    """
    free_vars = collect_free_vars(ast)
    power_set = [combo
                 for subset in (combinations(free_vars, r) for r in range(1 + len(free_vars)))
                 for combo in subset]
    for subset in power_set:
        if evaluate(ast, subset):
            return True
    return False


# noinspection PyPep8Naming
def proveFormula(formula: str):
    """
    Implements proveFormula according to grader.py
    >>> proveFormula('p')
    'S'
    >>> proveFormula('(NOT (NOT (NOT (NOT not))  )\t)')
    'S'
    >>> proveFormula('(IF p p)')
    'S'
    >>> proveFormula('(AND p (NOT p))')
    'U'
    >>> proveFormula('(OR p (NOT q))')
    'S'
    >>> proveFormula('(OR p (NOT q) q)')
    'S'
    >>> proveFormula('(OR (NOT q) q)')
    'S'
    >>> proveFormula('(AND (NOT q) q)')
    'U'
    >>> proveFormula('(AND (NOT q) q q)')
    'U'
    """
    return 'S' if determine_satisfiability(parse(formula)) else 'U'
