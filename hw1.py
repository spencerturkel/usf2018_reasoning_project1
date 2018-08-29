import re
from enum import Enum, auto
from typing import Union, Optional, Callable, TypeVar, NewType, Tuple, NamedTuple

# Defining the grammar in types
Formula = Union['FreeVar', 'SList']
FreeVar = NewType('FreeVar', str)
SList = Union['BinCall', 'NotCall']
BinCall = Tuple['BinOp', 'SList', 'SList']
NotCall = NewType('NotCall', SList)


class BinOp(Enum):
    AND = auto()
    IF = auto()
    OR = auto()


# Defining the type of Parsers
T = TypeVar('T')
ParseResult = NamedTuple('ParseResult', [('rest', str), ('value', T)])
Parser = Callable[[str], Optional[ParseResult[T]]]


def run_parser(p: Parser[T]) -> Callable[[str], Optional[str]]:
    def run(s: str) -> Optional[str]:
        result = p(s)
        if result is None:
            return None
        return result.value
    return run


def formula_parser(free_var_parser: Parser[Formula],
                   s_exp_parser: Parser[SList]) -> Parser[None]:
    return or_parser(free_var_parser, s_exp_parser)


U = TypeVar('U')


def or_parser(first_parser: Parser[T], second_parser: Parser[U]) -> Parser[Union[T, U]]:
    return lambda s: first_parser(s) or second_parser(s)


def free_var_parser() -> Parser[FreeVar]:
    regex = re.compile('[a-z0-9]+')

    def parse(s: str) -> Optional[ParseResult[FreeVar]]:
        pat = regex.match(s)
        return ParseResult(s, pat.group()) if pat is not None else None

    return parse


# noinspection PyPep8Naming
def proveFormula(formula: str) -> Union[int, str]:
    # TODO
    return formula
