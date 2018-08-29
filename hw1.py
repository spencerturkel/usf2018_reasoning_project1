import random
import re
from enum import Enum, auto
from string import digits, ascii_lowercase
from typing import Callable, Generic, NewType, Optional, Tuple, TypeVar, Union, AnyStr, Pattern
from unittest import TestCase

# Defining the grammar in types

Formula = Union['FreeVar', 'SList']
FreeVar = NewType('FreeVar', str)
SList = Union['BinCall', 'SList']
BinCall = Tuple['BinOp', 'SList', 'SList']


class BinOp(Enum):
    AND = auto()
    IF = auto()
    OR = auto()


# Defining the type of Parsers
T = TypeVar('T')
U = TypeVar('U')

class ParseResult(Generic[T]):
    def __init__(self, rest: str, value: T) -> None:
        self.rest = rest
        self.value = value


Parser = Callable[[str], Optional[ParseResult[T]]]


def map_parser(f: Callable[[T], U]) -> Callable[[Parser[T]], Parser[U]]:
    def mapped_parser(p: Parser[T]):
        def parse(s: str) -> Optional[U]:
            result = p(s)
            return f(result) if result is not None else None
        return parse
    return mapped_parser


run_parser = map_parser(lambda result: result.value)


def formula_parser(free_var_parser: Parser[Formula],
                   s_exp_parser: Parser[SList]) -> Parser[None]:
    return or_parser(free_var_parser, s_exp_parser)


def or_parser(first_parser: Parser[T], second_parser: Parser[U]) -> Parser[Union[T, U]]:
    return lambda s: first_parser(s) or second_parser(s)


def regex_parser(pattern: str) -> Parser[FreeVar]:
    regex = re.compile(pattern)

    def parse(s: str) -> Optional[ParseResult[FreeVar]]:
        pat = regex.match(s)
        if pat is None:
            return None
        match = pat.group()
        l = len(match)
        return ParseResult(s[l:], FreeVar(match))

    return parse


parse_free_var = regex_parser('[a-z0-9]+')


class FreeVarParserTests(TestCase):
    def test_valid_prefix_parsed(self):
        result = parse_free_var('hi123 hello')
        self.assertIsNotNone(result)
        self.assertEqual(' hello', result.rest)
        self.assertEqual(FreeVar('hi123'), result.value)

    def test_random_words_parse(self):
        for s in ''.join(random.choices(digits + ascii_lowercase, k=100)):
            with self.subTest():
                result = parse_free_var(s)
                self.assertIsNotNone(result)
                self.assertEqual('', result.rest)
                self.assertEqual(FreeVar(s), result.value)

    def test_parentheses_none(self):
        for s in ['(', ')']:
            with self.subTest():
                self.assertIsNone(parse_free_var(s))

    def test_whitespace_none(self):
        for s in [' ', '', '      ', '  ', '\n', '\t']:
            with self.subTest():
                self.assertIsNone(parse_free_var(s))


# noinspection PyPep8Naming
def proveFormula(formula: str) -> Union[int, str]:
    # TODO
    return formula
