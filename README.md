# proveFormula(F)
F is input string of an sexp of a PC formula

# Input
Undergrad may assume well-formed.

Arbitrary whitespace. Whitespace may be spaces, tabs, newlines, etc.
# S-exps
S-exp = freevar | list
freevar = [a-z0-9]+
list = (var-op S-exp\( S-exp\)\+) | (NOT S-exp) | (IF S-exp S-exp)
var-op = AND | OR
# Output
Returns either a string or an integer. NO OTHER OUTPUT.
- E if the formula was ill-formed (GRAD STUDENTS ONLY)
- T if the formula is a tautology
- U if the formula is unsatisfiable
- the number of variable bindings that are satisfiable
# Comments required
