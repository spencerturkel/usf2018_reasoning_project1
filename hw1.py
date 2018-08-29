from enum import Enum, auto
from typing import Union, Optional, Callable, TypeVar, NewType, Tuple

Formula = Union['FreeVar', 'SList']
FreeVar = NewType('FreeVar', str)
SList = Union['BinCall', 'NotCall']
BinCall = Tuple['BinOp', 'SList', 'SList']
NotCall = NewType('NotCall', SList)


class BinOp(Enum):
    AND = auto()
    IF = auto()
    OR = auto()


T = TypeVar('T')
Parser = Callable[[str], Optional[T]]


def formula_parser(free_var_parser: Parser[object], sexp_parser: Parser[None]) -> Parser[None]:
    def parse(s: str) -> Optional[str]:
        pass

    return parse


# noinspection PyPep8Naming
def proveFormula(formula: str) -> Union[int, str]:
    # TODO
    return formula
