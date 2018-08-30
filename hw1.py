import random
import re
from enum import Enum, auto
from string import digits, ascii_lowercase, ascii_uppercase
from typing import Callable, Generic, NewType, Optional, Tuple, TypeVar, Union, AnyStr, Pattern
from unittest import TestCase

T = TypeVar('T')
U = TypeVar('U')

# Defining the grammar in types

Formula = Union['FreeVar', 'SList']
FreeVar = NewType('FreeVar', str)
SList = Union['BinCall', 'NotCall']
NotCall = NewType('NotCall', SList)
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
        def parse(s: str) -> Optional[ParseResult[U]]:
            result = p(s)
            return ParseResult(result.rest, f(result.value)) if result is not None else None

        return parse

    return mapped_parser


class ParserTestCase(TestCase):
    def assertGoodParse(self, pinput: str, parser: Parser[T], rest: str, value: T):
        result = parser(pinput)
        self.assertIsNotNone(result)
        self.assertEqual(rest, result.rest)
        self.assertEqual(value, result.value)


class MapParserTests(ParserTestCase):
    random_input = str(random.random())
    random_value = random.random()

    def test_returns_identical_result(self):
        self.assertGoodParse(self.random_input,
                             map_parser(lambda x: x)(lambda s: ParseResult(s, self.random_value)),
                             self.random_input,
                             self.random_value)

    def test_returns_None_when_parser_returns_None(self):
        self.assertIsNone(map_parser(lambda x: x + 1)(lambda _: None)(''))

    def test_returns_transformed_result(self):
        self.assertGoodParse(self.random_input,
                             map_parser(lambda x: self.random_value)(lambda s: ParseResult(s, {})),
                             self.random_input,
                             self.random_value)


run_parser = map_parser(lambda result: result.value)


def choice_parser(first_parser: Parser[T], second_parser: Parser[U]) -> Parser[Union[T, U]]:
    return lambda s: first_parser(s) or second_parser(s)


def formula_parser(free_var_parser: Parser[Formula],
                   s_exp_parser: Parser[SList]) -> Parser[None]:
    return choice_parser(free_var_parser, s_exp_parser)


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


class FreeVarParserTests(ParserTestCase):
    def test_valid_prefix_parsed(self):
        self.assertGoodParse('hi123 hello', parse_free_var, ' hello', 'hi123')

    def test_random_words_good(self):
        for s in ''.join(random.choices(digits + ascii_lowercase, k=3)):
            with self.subTest():
                self.assertGoodParse(s, parse_free_var, '', s)

    def test_capitalized_words_bad(self):
        for s in ''.join(random.choices(digits + ascii_lowercase, k=3)):
            with self.subTest():
                self.assertIsNone(parse_free_var(random.choice(ascii_uppercase) + s))
                self.assertGoodParse(s, parse_free_var, '', s)

    def test_parentheses_none(self):
        for s in ['(', ')']:
            with self.subTest():
                self.assertIsNone(parse_free_var(s))

    def test_whitespace_none(self):
        for s in [' ', '', '      ', '  ', '\n', '\t']:
            with self.subTest():
                self.assertIsNone(parse_free_var(s))


def s_list_parser(bin_call_parser: Parser[BinCall],
                  not_call_parser: Parser[NotCall]) -> Parser[SList]:
    def parse(s: str) -> Optional[ParseResult[SList]]:
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
    def parse(inp: str) -> Optional[ParseResult[str]]:
        return ParseResult(inp[len(s):], s) if inp.startswith(s) else None

    return parse


class PrefixParserTests(TestCase):
    def test_good_parse(self):
        result = prefix_parser('abc')('abcdefg')
        self.assertIsNotNone(result)


class SListParserTests(TestCase):
    def test_starts_open_paren(self):
        self.assertIsNone(s_list_parser(failing_parser, failing_parser)('AND x y)'))

    def test_consumes_bin_call(self):
        result = s_list_parser(prefix_parser('a'), failing_parser)('(a) rest')
        self.assertIsNotNone(result)
        self.assertEqual('a', result.value)
        self.assertEqual(' rest', result.rest)

    def test_consumes_not_call(self):
        result = s_list_parser(failing_parser, map_parser(NotCall)(prefix_parser('a')))('(a) rest')
        self.assertIsNotNone(result)
        self.assertEqual('a', result.value)
        self.assertEqual(' rest', result.rest)


def bin_op_parser(op: BinOp) -> Parser[BinOp]:
    return map_parser(const(op))(prefix_parser(f'{op.name}'))


and_parser = bin_op_parser(BinOp.AND)
if_parser = bin_op_parser(BinOp.IF)
or_parser = bin_op_parser(BinOp.OR)


class AndParserTests(TestCase):
    def test_good_parse(self):
        result = and_parser('AND stuff')
        self.assertIsNotNone(result)
        self.assertEqual(' stuff', result.rest)
        self.assertEqual(BinOp.AND, result.value)


def apply_parser(func_parser: Parser[Callable[[T], U]], arg_parser: Parser[T]) -> Parser[U]:
    def parse_applied(s: str) -> Optional[ParseResult[U]]:
        func_result: Optional[ParseResult[Callable[[T], U]]] = func_parser(s)
        if func_result is None:
            return None
        rest: str = func_result.rest
        func: Callable[[T], U] = func_result.value
        return map_parser(func)(arg_parser)(rest)

    return parse_applied


class ApplyParserTests(ParserTestCase):
    def test_tupling(self):
        input = str(random.random())
        p1_result = str(random.random())
        p2_result = int(random.random())
        rest = str(random.random())

        def p1(s: str) -> Optional[ParseResult[str]]:
            self.assertEqual(input, s)
            return ParseResult(str(p2_result), p1_result)

        p2: Parser[int] = const(ParseResult(rest, p2_result))

        tupling_parser: Parser[Callable[[T], Tuple[str, T]]] = map_parser(lambda x: lambda y: (x, y))(p1)
        applied_parser = apply_parser(tupling_parser, p2)

        self.assertGoodParse(input, applied_parser, rest, (p1_result, p2_result))


def combine_parser(first: Parser[T], second: Parser[U]) -> Parser[U]:
    return apply_parser(map_parser(lambda _: lambda x: x)(first), second)


class CombineParserTests(ParserTestCase):
    def test_good_parse(self):
        self.assertGoodParse('hello world!',
                             combine_parser(prefix_parser('hello '), prefix_parser('world')),
                             '!',
                             'world')

    def test_bad_parse(self):
        self.assertIsNone(combine_parser(prefix_parser('hello '), prefix_parser('world'))('hi world!'))


whitespace_parser = regex_parser('\s+')


def bin_call_parser(formula_parser: Parser[Formula]) -> Parser[BinOp]:
    op_parser = choice_parser(and_parser, choice_parser(if_parser, or_parser))
    tupling_op_parser = map_parser(lambda op: lambda first_arg: lambda second_arg: (op, first_arg, second_arg))(
        op_parser)
    space_then_s_list_parser = combine_parser(whitespace_parser, formula_parser)
    return apply_parser(
        apply_parser(
            tupling_op_parser,
            space_then_s_list_parser),
        space_then_s_list_parser)


class BinCallParserTests(ParserTestCase):
    def test_good_parse(self):
        for op in BinOp:
            with self.subTest(op):
                self.assertGoodParse(f'{op.name} s_list \t\ns_listREST',
                                     bin_call_parser(prefix_parser('s_list')),
                                     'REST',
                                     (op, 's_list', 's_list'))

    def test_bad_parse(self):
        self.assertIsNone(bin_call_parser(prefix_parser('s_list'))('OP s_list \t\ns_list'))


parseFormula = run_parser(formula_parser(parse_free_var, s_list_parser(failing_parser, failing_parser)))


# noinspection PyPep8Naming
def proveFormula(formula: str) -> Union[int, str]:
    parseResult = parseFormula(formula)

    if parseResult is None:
        return 'E'

    # TODO
    return formula
