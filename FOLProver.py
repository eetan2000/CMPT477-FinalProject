import re
from typing import List

TOKEN_RE = re.compile(r"\s*(=>|->|<->|forall|exists|[A-Za-z_]\w*|[(),;.&|~]|\.|[^\s])")

# Start with tokenizing the formula
def tokenize(s: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(s) if t.strip() != ""]

# Recursively parse the formula. Start with top level expression and recursively parse the sub-level
# expressions to get a parse tree.
# EX. p(x) -> q(x)
# (
#     "implies",
#     ("pred", "p", [Term("x")]),
#     ("pred", "q", [Term("x")]),
# )
class Parser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def pop(self):
        t = self.peek()
        self.i += 1
        return t

    def parse_formula(self):
        return self.parse_iff()

    def parse_iff(self):
        left = self.parse_implies()
        while self.peek() == "<->":
            self.pop()
            right = self.parse_implies()
            left = ("iff", left, right)
        return left

    def parse_implies(self):
        left = self.parse_or()
        while self.peek() in ("->", "=>"):
            self.pop()
            right = self.parse_or()
            left = ("implies", left, right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.peek() == "|":
            self.pop()
            right = self.parse_and()
            left = ("or", left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.peek() in ("&", ";"):
            self.pop()
            right = self.parse_not()
            left = ("and", left, right)
        return left

    def parse_not(self):
        if self.peek() == "~":
            self.pop()
            return ("not", self.parse_not())
        if self.peek() in ("forall",):
            self.pop()
            var = self.pop()
            if self.peek() == ".":
                self.pop()
            body = self.parse_not()
            return ("forall", var, body)
        if self.peek() in ("exists",):
            self.pop()
            var = self.pop()
            if self.peek() == ".":
                self.pop()
            body = self.parse_not()
            return ("exists", var, body)
        return self.parse_atom()

    def parse_atom(self):
        t = self.peek()
        if t == "(":
            self.pop()
            f = self.parse_formula()
            if self.peek() == ")":
                self.pop()
            return f
        if re.match(r"[A-Za-z_]\w*", t):
            name = self.pop()
            if self.peek() == "(":
                self.pop()
                args = []
                if self.peek() != ")":
                    while True:
                        args.append(self.parse_term())
                        if self.peek() == ",":
                            self.pop()
                            continue
                        break
                if self.peek() == ")":
                    self.pop()
                return ("pred", name, args)
            return ("pred", name, [])
        raise SyntaxError(f"Unexpected token {t} at {self.i}")

    def parse_term(self):
        if self.peek() == "(":
            self.pop()
            t = self.parse_term()
            if self.peek() == ")":
                self.pop()
            return t
        tok = self.pop()
        if re.match(r"[A-Za-z_]\w*", tok):
            if self.peek() == "(":
                self.pop()
                args = []
                if self.peek() != ")":
                    while True:
                        args.append(self.parse_term())
                        if self.peek() == ",":
                            self.pop()
                            continue
                        break
                if self.peek() == ")":
                    self.pop()
                return Function(tok, args)
            return Term(tok, [])
        raise SyntaxError(f"Unexpected term token {tok} at {self.i}")


class Term:
    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []

    def is_var(self):
        return isinstance(self, Variable)

    def __repr__(self):
        if self.args:
            return f"{self.name}({', '.join(map(repr, self.args))})"
        return self.name

    def __eq__(self, other):
        return (
            isinstance(other, Term)
            and self.name == other.name
            and self.args == other.args
        )

    def __hash__(self):
        return hash((self.name, tuple(self.args)))


class Function(Term):
    pass


class Variable(Term):
    def __init__(self, name: str):
        super().__init__(name, [])

    def is_var(self):
        return True

### ============================= ###
#       Helper functions            #
### ============================= ###
# print F of, After which <stage>
def print_ast_stage(ast, stage):
    print("======= After " + stage + " =======")
    try:
        print(ast_to_str(ast))
    except Exception:
        print(f"  {repr(ast)}")
    print()


#&### for converting to Clausal Forms
### ===================================== ###
#       Step 2 - convert to PNF             #
#       (all quantifier at top level)       #
### ===================================== ###
# TODO: 
# Step 2.1 - eliminate iif <->
def eliminate_iff(ast):
    return ast

# TODO: 
# Step 2.2 - eliminate implies ->
def eliminate_implies(ast):
    return ast

# TODO: 
# Step 2.3 - push negation
def push_not(ast):
    return ast

# TODO: 
# Step 2.1 - standardize variables
def standardize_variables(ast):
    return ast

# Flow of convert to PNF
def to_PNF(ast):
    # Step 2.1 - eliminate iif
    ast2_1 = eliminate_iff(ast)
    print_ast_stage(ast2_1, "Step 2.1 - elimate iif <->")

    # Step 2.2 - eliminate implies
    ast2_2 = eliminate_implies(ast2_1)
    print_ast_stage(ast2_2, "Step 2.2 - eliminate implies ->")

    # Step 2.3 - push not close to terms
    ast2_3 = push_not(ast2_2)
    print_ast_stage(ast2_3, "Step 2.3 - push negation to terms")

    # Step 2.4 - standardize variables
    ast2_4 = standardize_variables(ast2_3)
    print_ast_stage(ast2_4, "Step 2.4 - standardize variable")
    return ast2_4


### ===================================== ###
#       Step 3 - skolemization              #
#       (remove existent quantifiers)       #
### ===================================== ###
def skolemize(ast):
    # TODO
    return ast


### ==================================== ###
#       Step 4 - drop quantifiers          #
#       (remove all quantifiers)           #
### ==================================== ###
def drop_universal(ast):
    if not isinstance(ast, tuple):
        return ast
    if ast[0] == "forall":
        return drop_universal(ast[2])
    if ast[0] in ("and", "or"):
        return (ast[0], drop_universal(ast[1]), drop_universal(ast[2]))
    if ast[0] == "not":
        return ("not", drop_universal(ast[1]))
    return ast


### ==================================== ###
#       Step 5 - convert to CNF            #
#       (distribute or over and)           #
### ==================================== ###
def distribute_or_over_and(ast):
    return ast


### ==================================== ###
#       Step 6 - to Clausal form           #
#       (set of clauses)                   #
### ==================================== ###
def to_clauses(ast):
    clauses = []
    return clauses


### ==================================== ###
#       the "Main" function flow           #
### ==================================== ###
### The "main" function
def prove_formula(input_str: str, max_steps: int = 5000):
    formula_text = extract_formula_input(input_str)
    # Parse
    try:
        ast = parse(formula_text)
        print_ast_stage(ast, "extract, Original Foumula")
    except Exception as e:
        return f"Parse error: {e}"   

    # to negation F
    ast1 = ("not", ast)
    
    # Step 2. to PNF (eliminate <->, ->, to NNF, rename)
    ast2 = to_PNF(ast1)
    print_ast_stage(ast2, "Step 2(result) - to PNF")

    # Step 3. skolemization
    ast3 = skolemize(ast2)
    print_ast_stage(ast3, "Step 3 - skolemization")

    # Step 4. drop universal quantifiers
    ast4 = drop_universal(ast3)
    print_ast_stage(ast4, "Step 4 - dropped quantifiers")

    # Step 5. to CNF (distribute or over and)
    ast5 = distribute_or_over_and(ast4)
    print_ast_stage(ast5, "Step 5 - to CNF")

    # Step 6. to Clausal form
    clauses = to_clauses(ast5)

    # TODO: Step 7. Resolution
    


### ==================================== ###
#       input's helper functions           #
### ==================================== ###
def extract_formula_input(s: str) -> str:
    if ":" in s:
        parts = s.split(":", 1)
        return parts[1].strip()
    return s.strip()


def parse(s: str):
    toks = tokenize(s)
    p = Parser(toks)
    return p.parse_formula()


def ast_to_str(ast):
    op = ast[0]
    if op == "pred":
        name = ast[1]
        args = ast[2]
        return (
            f"{name}({', '.join(term_to_str_obj(a) for a in args)})" if args else name
        )
    if op == "not":
        return f"~({ast_to_str(ast[1])})"
    if op == "and":
        return f"({ast_to_str(ast[1])} & {ast_to_str(ast[2])})"
    if op == "or":
        return f"({ast_to_str(ast[1])} | {ast_to_str(ast[2])})"
    if op == "implies":
        return f"({ast_to_str(ast[1])} -> {ast_to_str(ast[2])})"
    if op == "iff":
        return f"({ast_to_str(ast[1])} <-> {ast_to_str(ast[2])})"
    if op == "forall":
        return f"(forall {ast[1]}. {ast_to_str(ast[2])})"
    if op == "exists":
        return f"(exists {ast[1]}. {ast_to_str(ast[2])})"
    return str(ast)


def term_to_str_obj(t) -> str:
    if isinstance(t, Variable) or (isinstance(t, Term) and not t.args):
        return t.name
    if isinstance(t, Function) or (isinstance(t, Term) and t.args):
        return f"{t.name}({', '.join(term_to_str_obj(a) for a in t.args)})"
    return str(t)


if __name__ == "__main__":
    example = "F : (forall x.(p(x) & q(x))) -> (forall x.p(x) & forall x.q(x))"
    # example = "F:(forall x.(Person(x)) -> ~Likes(x,y))"
    result = prove_formula(example, max_steps=5000)
