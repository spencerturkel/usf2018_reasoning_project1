[![Build Status](https://travis-ci.com/spencerturkel/usf2018_reasoning_project1.svg?token=gm1zuwtz6yWqd9Rwapxf&branch=master)](https://travis-ci.com/spencerturkel/usf2018_reasoning_project1)

# What is this?
This project was the first project for our Computationally Modelling Reasoning course at the University of South Florida.

The assignment specification is detailed in the Slide JPG files.
As a brief overview, this project implements a [SAT Solver](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem) using the [DPLL algorithm](https://en.wikipedia.org/wiki/DPLL_algorithm).

The solver is implemented using Python 3.4, in the file [p1.py](https://github.com/spencerturkel/usf2018_reasoning_project1/blob/master/p1.py).

We used more than 100 unit tests (as [python doctests](https://pymotw.com/3/doctest/)) during the development of the project.

We integrated [Travis CI](https://travis-ci.com/) and [GitHub Pull Requests](https://github.com/spencerturkel/usf2018_reasoning_project1/pulls?utf8=%E2%9C%93&q=is%3Apr) to ensure that the master branch always passed all unit tests, and that all code was reviewed by at least one other person.

# proveFormula(F)
F is input string of an s-exp of a PC formula.

Must return within 60 seconds. *No other output*.
- `'S'` if the formula is satisfiable
- `'U'` if the formula is unsatisfiable

Must find satisfiability using DPLL algorithm.

# Formula Grammar
```
S-exp = ws freevar ws | ws list ws
ws = <empty> | <space> | <tab> | <newline> | <carriage return>
freevar = [a-z0-9]+
list = ( ws var-op ws S-exp ws S-exp-list ws )
     | ( ws NOT ws S-exp ws )
     | ( ws IF ws S-exp ws S-exp ws )
var-op = AND | OR
S-exp-list = S-exp | S-exp ws S-exp-list
```
# DPLL Algorithm
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
      * If `C` contains `p`, then remove the entire clause
      * If `C` contains `~p`, then remove `~p` from `C`
5. Exhaustively perform affirmative negation rule.
   1. Find a literal `p` which only appears all positive or all negative.
   2. Remove all clauses containing `p`.
6. Perform resolution to obtain two new formulas.
7. Recurse to step 2 with each new formula, returning first `True` result.
