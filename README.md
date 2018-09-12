# proveFormula(F)
F is input string of an sexp of a PC formula
Must implement DPLL algorithm
# DPLL
- Convert to CNF
- For each iteration:
-- Perform 1-literal rule
-- PErform affirmative negation rule
-- Perform resolution
-- If all clauses eliminated: return TRUE
-- If empty clause: return FALSE
-- Else, reiterate
# Input
May assume well-formed.

Arbitrary whitespace. Whitespace may be spaces, tabs, newlines, etc.
# S-exps
S-exp = freevar | list
freevar = [a-z0-9]+
list = (var-op S-exp\( S-exp\)\+) | (NOT S-exp) | (IF S-exp S-exp)
var-op = AND | OR
# Output
Must return within 60 seconds. NO OTHER OUTPUT.
- S if the formula is satisfiable
- U if the formula is unsatisfiable
# Comments required
# TODO
Update parser and evaluator for multiple args to AND/OR
