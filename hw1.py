import random
import re
from enum import Enum, auto
from string import digits, ascii_lowercase
from typing import Callable, Generic, NewType, Optional, Tuple, TypeVar, Union, AnyStr, Pattern
from unittest import TestCase

T = TypeVar('T')
U = TypeVar('U')

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
        value = pat.group()
        rest = s[len(value):]
        return ParseResult(rest, FreeVar(value))

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


def s_list_parser(bin_call_parser: Parser[BinCall],
                  not_call_parser: Parser[SList]) -> Parser[SList]:
    def parse(s: str) -> Optional[SList]:
        if not s.startswith('('):
            return None
        rest = s[1:]
        inner_result = bin_call_parser(rest) or not_call_parser(rest)
        if inner_result is None:
            return None
        return ParseResult(inner_result.rest[1:], inner_result.value)

    return parse


def const(val: T) -> Callable[[U], T]:
    return lambda _: val


failing_parser = const(None)


def prefix_parser(s: str) -> Parser[str]:
    def parse(inp: str) -> Optional[str]:
        return ParseResult(inp[len(s):], s) if inp.startswith(s) else None

    return parse


class SListParserTests(TestCase):
    def test_starts_open_paren(self):
        self.assertIsNone(s_list_parser(failing_parser, failing_parser)('AND x y)'))

    def test_consumes_bin_call(self):
        result = s_list_parser(prefix_parser('a'), failing_parser)('(a) rest')
        self.assertIsNotNone(result)
        self.assertEqual('a', result.value)
        self.assertEqual(' rest', result.rest)

    def test_consumes_not_call(self):
        result = s_list_parser(failing_parser, prefix_parser('a'))('(a) rest')
        self.assertIsNotNone(result)
        self.assertEqual('a', result.value)
        self.assertEqual(' rest', result.rest)


parseFormula = run_parser(formula_parser(parse_free_var, s_list_parser(failing_parser, failing_parser)))


# noinspection PyPep8Naming
def proveFormula(formula: str) -> Union[int, str]:
    parseResult = parseFormula(formula)

    if parseResult is None:
        return 'E'

# TODO
    return formula
