# CMPT477-FinalProject


## Project Overview

This project implements a **first-order logic theorem prover** based on **resolution**.

Features:
- Parses first-order formulas with quantifiers (∀, ∃), connectives (¬, ∧, ∨, →, ↔), functions and predicates.
- Transforms the input formula by:
  - eliminating → and ↔,
  - converting to Negation Normal Form (NNF),
  - standardizing variables apart,
  - performing Skolemization,
  - removing universal quantifiers,
  - converting to Conjunctive Normal Form (CNF) and extracting clauses.
- Applies first-order resolution with unification to the clause set of ¬F.
- Determines whether the original formula F is valid (by deriving the empty clause) and outputs the detailed proof steps.


## How to Run

### Requirements
- Python 3.9+  
- No external dependencies (only Python standard library)

### Run the built-in examples

python fol_prover_ui.py

## Input format: how to write formulas

You always type a single first-order formula as plain text.

### Quantifiers

Mathematical           | What to type
---------------------- | -------------
∀x. P(x)               | `forall x. P(x)`
∀x ∀y. Loves(x,y)      | `forall x,y. Loves(x,y)`
∃x. Student(x)         | `exists x. Student(x)`
∃x ∃y. R(x,y)          | `exists x,y. R(x,y)`

Notes:
- `forall` / `exists` can also be written as `all` / `some`.
- The dot after variables is optional, but recommended: `forall x. ...`
- If a name appears after a quantifier, it is treated as a **variable**, otherwise as a **constant**.


### Connectives

Mathematical         | What to type
-------------------- | -------------
¬F                   | `not F` or `~F` or `!F` or `¬F`
F ∧ G                | `F and G` or `F & G` or `F ∧ G`
F ∨ G                | `F or G` or `F \| G` or `F ∨ G`
F → G                | `F -> G` or `F => G` or `F → G`
F ↔ G                | `F <-> G` or `F <=> G` or `F ↔ G`

Operator precedence (from high to low):

`NOT` > `AND` > `OR` > `IMPLIES` (`->`) > `IFF` (`<->`)


### test cases

1. forall x. P(x) -> P(x)
   # expected: VALID

2. forall x. (p(x,x) -> exists y. p(x,y))
   # expected: VALID

3. (forall x. (P(x) and Q(x))) -> forall x. P(x)
   # expected: VALID

4. P -> Q
   # expected: NOT VALID (satisfiable)

5. forall x. P(x) -> Q(x)
   # parsed as: forall x. (P(x) -> Q(x))
   # expected: NOT VALID (satisfiable)

