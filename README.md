# proveFormula(F)
F is input string of an sexp of a PC formula
Must implement DPLL algorithm
# DPLL
[Wikipedia Article](https://en.wikipedia.org/wiki/DPLL_algorithm)

1. Convert to Conjunctive Normal Form (CNF)
  1. Transform IFs into disjunctions
  2. Push negations into literals using DeMorgan's
  3. Eliminate double negations
  4. Distribute disjunctions into conjunctions

  *Example*
  
  `(~p -> ~q) -> (p -> q)`
  1. `~(~~p + ~q) + (~p + q)`
  2. `(~~~p * ~~q) + (~p + q)`
  3. `(~p * q) + (~p + q)`
  4. `(~p + ~p + q) * (q + ~p + q)`
2. If all clauses eliminated, return `True`
3. If there is an empty clause, return `False`
4. Exhaustively perform 1-literal rule.
  1. Find a clause with a single literal `p`
  2. For every clause `C` containing `p` in any form:
  3. If `C` contains `p`, then remove the entire clause
  4. If `C` contains `~p`, then remove `~p` from `C`
5. Exhaustively perform affirmative negation rule.
  1. Find a literal `p` which only appears all `True` or all `False`.
  2. Remove all clauses containing `p`.
6. Perform resolution to obtain two new formulas.
7. Recurse to step 2 with each new formula, returning first `True` result.
# Input
May assume well-formed.

Arbitrary whitespace. Whitespace may be spaces, tabs, newlines, etc.
# S-exps
```
S-exp = freevar | list
freevar = [a-z0-9]+
list = (var-op S-exp\( S-exp\)\+) | (NOT S-exp) | (IF S-exp S-exp)
var-op = AND | OR
```
# Output
Must return within 60 seconds. NO OTHER OUTPUT.
- S if the formula is satisfiable
- U if the formula is unsatisfiable
# Comments required
# TODO
Update parser and evaluator for multiple args to AND/OR

