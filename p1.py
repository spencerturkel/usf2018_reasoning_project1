from enum import Enum
from itertools import combinations
from string import ascii_lowercase, digits, whitespace


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

    def __repr__(self):
        return 'Op.' + self.name


ascii_lowercase_plus_digits = ascii_lowercase + digits


def parse(formula: str):
    """
    Parses a formula string into an AST tuple representation using Op.

    >>> parse('(AND (IF p q) (NOT r))')
    (Op.AND, (Op.IF, 'p', 'q'), (Op.NOT, 'r'))
    >>> parse('(OR p (NOT q))')
    (Op.OR, 'p', (Op.NOT, 'q'))
    >>> parse(' ( OR p ( NOT q ) ) ')
    (Op.OR, 'p', (Op.NOT, 'q'))
    >>> parse('(AND (IF p q) (NOT r) (OR p q r))') # AND/OR take 2+ args
    (Op.AND, (Op.IF, 'p', 'q'), (Op.NOT, 'r'), (Op.OR, 'p', 'q', 'r'))
    >>> parse('q1')
    'q1'
    >>> parse(' ( OR p ( NOT q ) ) ')
    (Op.OR, 'p', (Op.NOT, 'q'))
    """
    # Implemented as a recursive-descent parser.
    index = 0  # Current view into formula
    length = len(formula)

    def consume_whitespace():
        """Increases the index while the current token is whitespace."""
        nonlocal formula, index
        while formula[index] in whitespace:
            index += 1

    def parse_form():
        """Parses a formula."""
        nonlocal formula, index

        ch = formula[index]

        if ch == '(':
            # parsing a call
            index += 1  # consume '('
            consume_whitespace()
            result = parse_list()
            consume_whitespace()
            index += 1  # consume ')'
            return result

        # parsing a literal
        literal = []
        while index < length and formula[index] in ascii_lowercase_plus_digits:
            literal.append(formula[index])
            index += 1
        return ''.join(literal)

    def parse_list():
        """Parses the contents of a parenthesized s-exp list"""
        nonlocal formula, index

        if formula.startswith(Op.NOT.name, index):
            index += 3  # len('NOT')
            consume_whitespace()
            first_arg = parse_form()
            return Op.NOT, first_arg

        if formula.startswith(Op.IF.name, index):
            index += 2  # len('IF')
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

    consume_whitespace()
    return parse_form()


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
    :return: True if satisfiable, False if unsatisfiable

    >>> determine_satisfiability('x')
    True
    >>> determine_satisfiability((Op.NOT, 'x'))
    True
    >>> determine_satisfiability((Op.OR, 'x', (Op.NOT, 'x')))
    True
    >>> determine_satisfiability((Op.AND, 'x', (Op.NOT, 'x')))
    False
    >>> determine_satisfiability((Op.OR, 'x', 'y'))
    True
    >>> determine_satisfiability((Op.OR, 'x', (Op.OR, 'y', 'z')))
    True
    >>> determine_satisfiability((Op.NOT, (Op.OR, 'x', (Op.OR, 'y', 'z'))))
    True
    """
    free_vars = collect_free_vars(ast)
    power_set = [combo
                 for subset in (combinations(free_vars, r) for r in range(1 + len(free_vars)))
                 for combo in subset]
    for subset in power_set:
        if evaluate(ast, subset):
            return True
    return False


def transform_ifs(node):
    """
    >>> transform_ifs((Op.IF, 'x', 'y'))
    (Op.OR, (Op.NOT, 'x'), 'y')
    >>> transform_ifs((Op.IF, (Op.IF, (Op.NOT, 'x'), (Op.NOT, 'y')), (Op.IF, 'x', 'y')))
    (Op.OR, (Op.NOT, (Op.OR, (Op.NOT, (Op.NOT, 'x')), (Op.NOT, 'y'))), (Op.NOT, 'x'), 'y')
    """
    if isinstance(node, str):
        return node
    transformed_args = tuple(map(transform_ifs, node[1:]))
    if node[0] == Op.IF:
        second_arg = transformed_args[1]
        args_after_first = second_arg[1:] if second_arg[0] == Op.OR else (second_arg,)  # flatten nested ORs
        return (Op.OR, (Op.NOT, transformed_args[0])) + args_after_first
    return (node[0],) + transformed_args


def push_negations_to_literals(node):
    """
    >>> push_negations_to_literals((Op.OR, (Op.NOT, 'x'), 'y'))
    (Op.OR, (Op.NOT, 'x'), 'y')
    >>> push_negations_to_literals((Op.NOT, (Op.OR, 'x', 'y')))
    (Op.AND, (Op.NOT, 'x'), (Op.NOT, 'y'))
    >>> push_negations_to_literals((Op.NOT, (Op.AND, 'x', 'y')))
    (Op.OR, (Op.NOT, 'x'), (Op.NOT, 'y'))
    >>> push_negations_to_literals((Op.NOT, (Op.AND, (Op.NOT, (Op.OR, 'x', 'y')), 'z')))
    (Op.OR, (Op.NOT, (Op.NOT, 'x')), (Op.NOT, (Op.NOT, 'y')), (Op.NOT, 'z'))
    >>> push_negations_to_literals((Op.AND, 'x', (Op.NOT, (Op.OR, 'x', 'y')), 'z'))
    (Op.AND, 'x', (Op.NOT, 'x'), (Op.NOT, 'y'), 'z')
    >>> push_negations_to_literals((Op.OR, 'x', (Op.NOT, (Op.AND, 'x', 'y')), 'z'))
    (Op.OR, 'x', (Op.NOT, 'x'), (Op.NOT, 'y'), 'z')
    """
    if isinstance(node, str):
        return node
    for op in [Op.AND, Op.OR]:
        if node[0] == op:
            call = (op,)
            for arg in (map(push_negations_to_literals, node[1:])):
                if arg[0] == op:
                    call += arg[1:]  # flatten nested op
                else:
                    call += (arg,)
            return call
    # node[0] == Op.NOT
    arg = node[1]
    if isinstance(arg, str) or arg[0] == Op.NOT:
        return Op.NOT, push_negations_to_literals(arg)
    op = Op.AND if arg[0] == Op.OR else Op.OR
    call = (op,)
    for arg in map(push_negations_to_literals, node[1:]):
        sub_args = tuple(map(lambda x: (Op.NOT, x), arg[1:]))
        call += sub_args
    return call


def remove_double_negations(ast):
    """
    :param ast: Result from parse()
    :return: An AST where sub-nodes of the form (Op.NOT, (Op.NOT, x)) are replaced with x
    >>> remove_double_negations('x')
    'x'
    >>> remove_double_negations((Op.NOT, 'x'))
    (Op.NOT, 'x')
    >>> remove_double_negations((Op.NOT, (Op.NOT, 'x')))
    'x'
    >>> remove_double_negations((Op.NOT, (Op.NOT, (Op.NOT, 'x'))))
    (Op.NOT, 'x')
    >>> remove_double_negations((Op.AND, 'x', 'y'))
    (Op.AND, 'x', 'y')
    """
    if isinstance(ast, str):
        return ast
    op, arg, *_ = ast
    if op == Op.NOT and arg[0] == Op.NOT:
        return remove_double_negations(arg[1])
    return (op,) + tuple(map(remove_double_negations, ast[1:]))


def distribute_disjunctions(ast):
    """
    >>> distribute_disjunctions('x')
    'x'
    >>> distribute_disjunctions((Op.NOT, 'x'))
    (Op.NOT, 'x')
    >>> distribute_disjunctions((Op.AND, 'x', 'y'))
    (Op.AND, 'x', 'y')
    >>> distribute_disjunctions((Op.OR, 'x', 'y'))
    (Op.OR, 'x', 'y')
    >>> distribute_disjunctions((Op.AND, (Op.AND, 'x', 'y'), 'z'))
    (Op.AND, 'x', 'y', 'z')
    >>> distribute_disjunctions((Op.OR, (Op.OR, 'x', 'y'), 'z'))
    (Op.OR, 'x', 'y', 'z')
    >>> distribute_disjunctions((Op.AND, (Op.OR, 'x', 'y'), 'z'))
    (Op.AND, (Op.OR, 'x', 'y'), 'z')
    >>> distribute_disjunctions((Op.AND, (Op.OR, 'x', 'y'), (Op.OR, 'x', 'z')))
    (Op.AND, (Op.OR, 'x', 'y'), (Op.OR, 'x', 'z'))
    >>> distribute_disjunctions((Op.AND, 'p', (Op.OR, 'q', 'r'), 's'))
    (Op.AND, 'p', (Op.OR, 'q', 'r'), 's')
    >>> distribute_disjunctions((Op.OR, 'p', (Op.AND, 'q', 'r'), 's'))
    (Op.AND, (Op.OR, 'p', 'q', 's'), (Op.OR, 'p', 'r', 's'))
    >>> distribute_disjunctions((Op.OR, (Op.AND, 'q', 'r'), 'p', 's'))
    (Op.AND, (Op.OR, 'q', 'p', 's'), (Op.OR, 'r', 'p', 's'))
    >>> distribute_disjunctions((Op.OR, 's', 'p', (Op.AND, 'q', 'r')))
    (Op.AND, (Op.OR, 's', 'p', 'q'), (Op.OR, 's', 'p', 'r'))
    """
    if isinstance(ast, str):
        return ast
    op, *args = ast
    args = map(distribute_disjunctions, args)
    if op == Op.NOT:
        return (op,) + tuple(args)

    disjunction_lists = [[]]
    for arg in args:
        arg_op = arg[0]
        if isinstance(arg, str) or arg_op == Op.NOT or op == Op.AND and arg_op == Op.OR:
            for l in disjunction_lists:
                l.append(arg)
        elif (Op.AND == op == arg_op) or (Op.OR == op == arg_op):
            for l in disjunction_lists:
                l.extend(arg[1:])
        else:  # op == Op.OR and arg_op == Op.AND:
            disjunction_lists = [l + [x] for x in arg[1:] for l in disjunction_lists]
    if len(disjunction_lists) == 1:
        return (op,) + tuple(disjunction_lists[0])
    return (Op.AND,) + tuple(map(lambda x: tuple([Op.OR] + x), disjunction_lists))


def convert_to_cnf(ast):
    """
    Transforms the parsed AST into Conjunctive Normal Form.
    :param ast: Result from parse()
    :return: An AST of the form (Op.AND, ...), where the rest of the AST does NOT have any Op.AND values.

    >>> convert_to_cnf('x')
    'x'
    >>> convert_to_cnf((Op.AND, 'x', 'y'))
    (Op.AND, 'x', 'y')
    >>> convert_to_cnf((Op.OR, 'x', 'y'))
    (Op.OR, 'x', 'y')
    >>> convert_to_cnf((Op.OR, (Op.AND, 'x', 'y'), (Op.AND, 'y', 'z')))
    (Op.AND, (Op.OR, 'x', 'y'), (Op.OR, 'y', 'y'), (Op.OR, 'x', 'z'), (Op.OR, 'y', 'z'))
    >>> convert_to_cnf((Op.IF, (Op.IF, (Op.NOT, 'p'), (Op.NOT, 'q')), (Op.IF, 'p', 'q')))
    (Op.AND, (Op.OR, (Op.NOT, 'p'), (Op.NOT, 'p'), 'q'), (Op.OR, 'q', (Op.NOT, 'p'), 'q'))
    >>> convert_to_cnf((Op.IF, (Op.IF, (Op.NOT, 'q'), (Op.NOT, 'p')), (Op.IF, 'p', 'q')))
    (Op.AND, (Op.OR, (Op.NOT, 'q'), (Op.NOT, 'p'), 'q'), (Op.OR, 'p', (Op.NOT, 'p'), 'q'))
    """
    ast = transform_ifs(ast)
    ast = push_negations_to_literals(ast)
    ast = remove_double_negations(ast)
    ast = distribute_disjunctions(ast)
    return ast


def dpll(ast):
    """
    Runs the DPLL algorithm on the parsed AST.

    This function should always agree with determine_satisfiability(ast).

    :param ast: Result from parse()
    :return: True if satisfiable, False if unsatisfiable

    >>> dpll('x')
    True
    >>> dpll((Op.NOT, 'x'))
    True
    >>> dpll((Op.OR, 'x', (Op.NOT, 'x')))
    True
    >>> dpll((Op.AND, 'x', (Op.NOT, 'x')))
    False
    >>> dpll((Op.OR, 'x', 'y'))
    True
    >>> dpll((Op.OR, 'x', (Op.OR, 'y', 'z')))
    True
    >>> dpll((Op.NOT, (Op.OR, 'x', (Op.OR, 'y', 'z'))))
    True
    """
    ast = convert_to_cnf(ast)
    # TODO
    return determine_satisfiability(ast)

def resolution(formula):
    """
    This function performs resolution on a parsed AST
    returns returns a satisfiable formula if only one variable left EX: (OR a)
    returns returns a unsatisfiable formula if there are no variables left
    returns resolved Formula otherwise

    >>> resolution(AND (OR a b c) (OR (NOT a)  )     )
    True
    >>> resolution (AND (OR a b c) (OR (NOT a)) (OR  d e f)  )
    True
    >>> resolution (AND (OR a b c) (OR (NOT a)) (OR  d e f) (OR (NOT d) (NOT e) (NOT f)) )
    False
    >>> resolution (AND (OR a b c) (OR (NOT a) (NOT b) (NOT c)) )
    False
    >>> resolution (AND (OR a b c) (OR (NOT a) (NOT b) ) )
    True



    """
    if len(formula) == 1:
        return formula
    i = 0
    if formula[i] != Op.AND:
        return formula

    i+=1
    j=0
    somethingResolved = False
    resolvedFormula = ((Op.AND,))

    # loops through entire formula looking for negation of current variable
    while i < len(formula):
        if(len(formula[i])) == 0:
            formula = (Op.AND, 'x', (Op.NOT, 'x'))
            print(formula, "1")
            return formula
        while j < len(formula[i]):
            var = formula[i][j]
            if var is Op.OR and Op.AND and Op.IF:
                j+=1
                if j >= len(formula[i]):
                    print(formula, "1")
                    return formula
                elif var == Op.NOT:
                    j+=1
                    resolveVar = '~' + formula[i][j]
                elif len(formula[i][j]) == 2:
                    resolveVar = '~' + formula[i][j][1]
                else:
                    resolveVar = formula[i][j]
            else:
                if var == Op.NOT:
                    j+=1
                    resolveVar = '~' + formula[i][j]
                elif len(formula[i][j]) == 2:
                    resolveVar = '~' + formula[i][j][1]
                else:
                    resolveVar = formula[i][j]

            m = i +1
            n = 0

            # loops through formula to find negation of resolveVar
            while m < len(formula):
                n=0
                while n < len(formula[m]):
                    var2 = formula[m][n]
                    if len(resolveVar) == 2:
                        if var2 is Op.OR and Op.AND and Op.IF:
                            pass
                        else:
                            if var2 == resolveVar[1]:
                                resolvedFormula = resolveMatched(formula, i, j, m, n, resolvedFormula)
                                somethingResolved = True
                    else:
                        if var2 is Op.OR and Op.AND and Op.IF:
                            pass
                        elif var2 == Op.NOT:
                            n+=1
                            var2 = formula[m][n]
                            if var2 == resolveVar:
                                resolvedFormula = resolveMatched(formula, i, j, m, n, resolvedFormula)
                                somethingResolved = True
                        elif len(var2) == 2:
                            if var2[1] == resolveVar:
                                resolvedFormula = resolveMatched(formula,i,j,m,n,resolvedFormula)
                                somethingResolved = True
                        if resolvedFormula == 'z':
                            formula = (Op.AND, 'x', (Op.NOT, 'x'))
                            print(formula,"1")
                            return formula
                    n+=1
                m+=1
            j+=1

        # if current list wasn't resolved, add it to resolvedFormula
        if somethingResolved == False:
            resolvedFormula = (resolvedFormula) + ((formula[i],))
        i+=1
        j=0



    # if list = (AND (OR variables v)) remove initial AND
    if formula[0] is Op.AND and len(formula) ==2:
        formula = formula[1]
    # performs resolve on new formula if something was resolved
    elif somethingResolved:
        formula = resolution(resolvedFormula)
    else:
        a = 0
        while a < len(formula):
            if formula[a] is not Op.AND:
                if formula[a] is not Op.OR:
                    if len(formula[a]) == 1:
                        formula = (Op.NOT, 'x')
                        print(formula, "1")
                        return formula
            a+=1
    print(formula, "1")
    return formula
def resolveMatched(formula, i,j,m,n, resolvedFormula):
    # this function deletes clauses that were resolved and returns a resolvedFormula
    z = 0
    tempFormula = ()
    while z < len(formula[i]):
        if z is not j:
            tempFormula = tempFormula + (formula[i][z],)
        z += 1
    z = 1
    while z < len(formula[m]):
        if z is not n:
            tempFormula = tempFormula + (formula[m][z],)
        z += 1
    a = 0

    while a < len(tempFormula):
        b = a +1
        if type(tempFormula[a]) is Op:
            pass
        else:
            while b < len(tempFormula):
                if tempFormula[a] == tempFormula[b]:
                    tempFormula = list(tempFormula)
                    del tempFormula[a]
                    a = 0
                b+=1
        a+=1
    a = 0

    while a < len(tempFormula):
        b = a+1
        while b < len(tempFormula):
            if type(tempFormula[a]) is Op.NOT:
                pass
            elif type(tempFormula[a]) is Op:
                pass
            elif len(tempFormula[a]) == 2:
                if tempFormula[a][1] == tempFormula[b]:
                    tempFormula = list(tempFormula)
                    del tempFormula[a]
                    del tempFormula[b-1]
                    a-=2
            else:
                if len(tempFormula[b]) == 2:
                    if tempFormula[a] == tempFormula[b][1]:
                        tempFormula = list(tempFormula)
                        del tempFormula[a]
                        del tempFormula[b-1]
                        a-=2
            b+=1
        a+=1
    a=0

    if tempFormula[0] == Op.OR:
        if len(tempFormula) ==2:
            tempFormula = (tempFormula[1])
    while a < len(tempFormula):
        if len(tempFormula) == 1 and tempFormula[0] is Op.OR:
            return 'z'
        elif len(tempFormula) == 1 and tempFormula[0] is Op.NOT:
            return 'z'
        a+=1
    tempFormula = tuple(tempFormula)
    resolvedFormula = (resolvedFormula) + ((tempFormula,))

    return resolvedFormula


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
    return 'S' if dpll(parse(formula)) else 'U'
