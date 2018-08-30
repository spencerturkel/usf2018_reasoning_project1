import string
from enum import Enum, auto
from string import ascii_lowercase, digits
from typing import Union
from unittest import TestCase


# Formula Grammar
###########################################################
# F = V | (L)
# V = aV' | bV' | ... | zV' | 0V' | 1V' | ... | 9V'
# V' = aV' | bV' | ... | zV' | 0V' | 1V' | ... | 9V' | ''
# L = O F F | 'NOT' F
# O = 'IF' | 'AND' | 'OR'
###########################################################

class Op(Enum):
    AND = auto()
    IF = auto()
    NOT = auto()
    OR = auto()


def parse(formula: str):
    """
    Parses a formula string into a tuple representation using Op.

    Implemented as a recursive-descent parser.
    """
    length = len(formula)
    index = 0  # current view into formula

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
            lex()
            result = parse_list()
            consume_whitespace()
            if index < length and formula[index] == ')':
                lex()
                return result
            raise ValueError('Unclosed form')
        if ch in (ascii_lowercase + digits):
            word = []
            while index < length and formula[index] in (ascii_lowercase + digits):
                word.append(formula[index])
                index += 1
            return ''.join(word)

    def lex():
        """Increments the index then consumes whitespace."""
        nonlocal index
        index += 1
        consume_whitespace()

    def parse_list():
        """Parses the contents of a parenthesized s-exp list"""
        nonlocal formula, index
        for op in Op:
            if formula.startswith(op.name, index):
                index += len(op.name)
                lex()
                first_arg = parse_form()
                if op == Op.NOT:
                    consume_whitespace()
                    return op, first_arg
                lex()
                second_arg = parse_form()
                return op, first_arg, second_arg

    consume_whitespace()
    try:
        result = parse_form()
        return None if index < length else result
    except ValueError:
        return None


class ParseTests(TestCase):
    def test_good_one(self):
        self.assertEqual('xy', parse('xy'))

    def test_good_two(self):
        self.assertEqual((Op.NOT, 'p'), parse('(NOT p)'))

    def test_good_three(self):
        self.assertEqual((Op.IF, 'p', 'q'), parse('\t(\nIF p q\t)'))

    def test_good_four(self):
        self.assertEqual((Op.NOT, (Op.NOT, (Op.NOT, (Op.NOT, 'not')))),
                         parse('(NOT (NOT (NOT (NOT not))  )		)'))

    def test_bad_extra_parentheses(self):
        self.assertIsNone(parse('((NOT p))'))

    def test_bad_unbalanced_left(self):
        self.assertIsNone(parse('((NOT (NOT (NOT (NOT not))  )		)'))

    def test_bad_unbalanced_right(self):
        self.assertIsNone(parse('(NOT (NOT (NOT (NOT not))  )		))'))

    def test_bad_op_capitalization(self):
        self.assertIsNone(parse('(NoT x)'))

    def test_bad_var_capitalization(self):
        self.assertIsNone(parse('(NOT X)'))

    def test_bad_spelling(self):
        self.assertIsNone(parse('(NT x)'))


# noinspection PyPep8Naming
def proveFormula(formula: str) -> Union[int, str]:
    parseResult = parse(formula)

    if parseResult is None:
        return 'E'

    # TODO
    return formula
