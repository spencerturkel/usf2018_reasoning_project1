# proveFormula(F)
F is input string of an sexp of a PC formula
Must implement DPLL algorithm
# DPLL
- Convert to CNF
- For each iteration:
<ul>
<li> Perform 1-literal rule</li>
<ul>
  <li>Find a clause with a single literal p</li>
  <li> For every clause C containing p in any form:</li>
  <li>If C contains p, then remove the entire clause</li>
  <li>If C contains -p, then remove -p from C</li>
</ul>
  <li> Perform affirmative negation rule </li>
  <li> Perform resolution </li>
  <li> If all clauses eliminated: return TRUE </li>
  <li> If empty clause: return FALSE </li>
  <li> Else, reiterate </li>
</ul>
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

