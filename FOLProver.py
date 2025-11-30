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


def prove_formula(input_str: str, max_steps: int = 5000):
    formula_text = extract_formula_input(input_str)

    # 1. Parse
    try:
        ast = parse(formula_text)
    except Exception as e:
        return f"Parse error: {e}"

    print("Original Formula:")
    print_formula(ast)

    # 2. Eliminate -> and <-> from the formula
    ast_no_imp = eliminate_implies_iff(ast)
    print("After eliminating -> and <-> :")
    print_formula(ast_no_imp)

    # 3. Convert to Negation Normal Form (NNF)
    ast_nnf = to_nnf(ast_no_imp)
    print("NNF:")
    print_formula(ast_nnf)

    # 4. Skolemization
    ast_sko = skolemize(ast_nnf)
    print("After Skolemization:")
    print_formula(ast_sko)

    # 5. Drop all quantifiers
    ast_matrix = drop_quantifiers(ast_sko)
    print("After dropping quantifiers:")
    print_formula(ast_matrix)

    # 6. Convert the to CNF
    ast_cnf = to_cnf(ast_matrix)
    print("CNF (AST):")
    print_formula(ast_cnf)

    return ast_cnf


def extract_formula_input(s: str) -> str:
    if ":" in s:
        parts = s.split(":", 1)
        return parts[1].strip()
    return s.strip()


def parse(s: str):
    toks = tokenize(s)
    p = Parser(toks)
    return p.parse_formula()

def print_formula(ast):
    try:
        print(ast_to_str(ast))
    except Exception:
        print(repr(ast))
    print()

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

def eliminate_implies_iff(ast):
    op = ast[0]
    if op == "pred":
        return ast
    if op == "not":
        return ("not", eliminate_implies_iff(ast[1]))

    if op == "and":
        return ("and",
                eliminate_implies_iff(ast[1]),
                eliminate_implies_iff(ast[2]))

    if op == "or":
        return ("or",
                eliminate_implies_iff(ast[1]),
                eliminate_implies_iff(ast[2]))

    if op == "forall":
        var, body = ast[1], ast[2]
        return ("forall", var, eliminate_implies_iff(body))

    if op == "exists":
        var, body = ast[1], ast[2]
        return ("exists", var, eliminate_implies_iff(body))

    if op == "implies":
        A = eliminate_implies_iff(ast[1])
        B = eliminate_implies_iff(ast[2])
        return ("or", ("not", A), B)

    if op == "iff":
        A = eliminate_implies_iff(ast[1])
        B = eliminate_implies_iff(ast[2])
        left_part = ("or", ("not", A), B)
        right_part = ("or", ("not", B), A)
        return ("and", left_part, right_part)


def to_nnf(ast):
    op = ast[0]
    if op == "pred":
        return ast
    if op == "and":
        return ("and", to_nnf(ast[1]), to_nnf(ast[2]))
    if op == "or":
        return ("or", to_nnf(ast[1]), to_nnf(ast[2]))
    if op == "forall":
        var, body = ast[1], ast[2]
        return ("forall", var, to_nnf(body))
    if op == "exists":
        var, body = ast[1], ast[2]
        return ("exists", var, to_nnf(body))
    if op == "not":
        sub = ast[1]
        sop = sub[0]
        # ~~A  =  A
        if sop == "not":
            return to_nnf(sub[1])

        # ~(A & B)  =  ~A | ~B
        if sop == "and":
            A, B = sub[1], sub[2]
            return ("or", to_nnf(("not", A)), to_nnf(("not", B)))

        # ~(A | B)  =  ~A & ~B
        if sop == "or":
            A, B = sub[1], sub[2]
            return ("and", to_nnf(("not", A)), to_nnf(("not", B)))

        # ~(forall x. A)  ≡  exists x. ~A
        if sop == "forall":
            var, body = sub[1], sub[2]
            return ("exists", var, to_nnf(("not", body)))

        # ~(exists x. A)  ≡  forall x. ~A
        if sop == "exists":
            var, body = sub[1], sub[2]
            return ("forall", var, to_nnf(("not", body)))

        raise ValueError(f"Unknown op in to_nnf: {op}")
    raise ValueError(f"Unknown AST node in to_nnf: {ast}")



def subst_in_term(t: Term, varname: str, repl: Term) -> Term:
    if isinstance(t, Term):
        if not t.args and t.name == varname:
            return repl
        if t.args:
            new_args = [subst_in_term(a, varname, repl) for a in t.args]
            if isinstance(t, Function):
                return Function(t.name, new_args)
            else:
                return Term(t.name, new_args)
    return t


def subst_in_ast(ast, varname: str, repl: Term):
    op = ast[0]
    if op == "pred":
        name, args = ast[1], ast[2]
        new_args = [subst_in_term(a, varname, repl) for a in args]
        return ("pred", name, new_args)

    if op in ("and", "or"):
        return (op,
                subst_in_ast(ast[1], varname, repl),
                subst_in_ast(ast[2], varname, repl))

    if op == "not":
        return ("not", subst_in_ast(ast[1], varname, repl))

    if op in ("forall", "exists"):
        var, body = ast[1], ast[2]
        return (op, var, subst_in_ast(body, varname, repl))

    return ast




def drop_quantifiers(ast):
    op = ast[0]
    if op == "pred":
        return ast
    if op == "not":
        return ("not", drop_quantifiers(ast[1]))
    if op in ("and", "or"):
        return (op,
                drop_quantifiers(ast[1]),
                drop_quantifiers(ast[2]))
    if op in ("forall", "exists"):
        return drop_quantifiers(ast[2])
    raise ValueError(f"Unknown op in drop_quantifiers: {op}")




def skolemize(ast):
    counter = [0]

    def fresh_sk_name():
        counter[0] += 1
        return f"SK{counter[0]}"

    skolem_cache = {}

    def sko(node, univ_vars: List[str]):
        op = node[0]

        if op == "pred":
            return node

        if op == "not":

            return ("not", sko(node[1], univ_vars))

        if op in ("and", "or"):

            left = sko(node[1], univ_vars)
            right = sko(node[2], univ_vars)
            return (op, left, right)

        if op == "forall":
            var, body = node[1], node[2]

            new_univ = univ_vars + [var]
            return ("forall", var, sko(body, new_univ))

        if op == "exists":
            var, body = node[1], node[2]

            if var in skolem_cache:
                sk_name = skolem_cache[var]
            else:
                sk_name = fresh_sk_name()
                skolem_cache[var] = sk_name

            if univ_vars:
                args = [Variable(v) for v in univ_vars]
                sk_term = Function(sk_name, args)
            else:
                sk_term = Variable(sk_name)
            body2 = subst_in_ast(body, var, sk_term)
            return sko(body2, univ_vars)

        raise ValueError(f"Unknown op in skolemize: {op}")

    return sko(ast, [])



def distribute_or(a, b):
    # (A1 & A2) | B  ≡  (A1 | B) & (A2 | B)
    if isinstance(a, tuple) and a[0] == "and":
        return ("and",
                distribute_or(a[1], b),
                distribute_or(a[2], b))

    # A | (B1 & B2)  ≡  (A | B1) & (A | B2)
    if isinstance(b, tuple) and b[0] == "and":
        return ("and",
                distribute_or(a, b[1]),
                distribute_or(a, b[2]))

    return ("or", a, b)


def to_cnf(ast):
    op = ast[0]

    if op == "pred":
        return ast
    if op == "not":

        sub = ast[1]
        if sub[0] == "pred":
            return ast

    if op == "and":
        # CNF(A & B) = CNF(A) & CNF(B)
        left = to_cnf(ast[1])
        right = to_cnf(ast[2])
        return ("and", left, right)

    if op == "or":
        # CNF(A | B) = distribute_or(CNF(A), CNF(B))
        left = to_cnf(ast[1])
        right = to_cnf(ast[2])
        return distribute_or(left, right)

    raise ValueError(f"Unknown op in to_cnf: {op}")




if __name__ == "__main__":
    example = "F : (forall x.(p(x) & q(x))) -> (forall x.p(x) & forall x.q(x))"
    example = "F:(forall x.(Person(x)) -> ~Likes(x,y))"
    result = prove_formula(example, max_steps=5000)


