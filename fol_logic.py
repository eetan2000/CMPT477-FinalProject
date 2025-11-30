from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Set, Iterable


class Term:
    pass


@dataclass(frozen=True)
class Var(Term):
    name: str


@dataclass(frozen=True)
class Const(Term):
    name: str


@dataclass(frozen=True)
class Func(Term):
    name: str
    args: Tuple[Term, ...]


class Formula:
    pass


@dataclass(frozen=True)
class Pred(Formula):
    name: str
    args: Tuple[Term, ...]


@dataclass(frozen=True)
class Not(Formula):
    sub: Formula


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Iff(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class ForAll(Formula):
    var: str
    body: Formula


@dataclass(frozen=True)
class Exists(Formula):
    var: str
    body: Formula


@dataclass(frozen=True)
class Literal:
    pred: str
    args: Tuple[Term, ...]
    positive: bool = True


Clause = frozenset[Literal]


def term_to_str(t: Term) -> str:
    if isinstance(t, Var):
        return t.name
    if isinstance(t, Const):
        return t.name
    if isinstance(t, Func):
        if not t.args:
            return t.name
        return f"{t.name}(" + ", ".join(term_to_str(a) for a in t.args) + ")"
    return "<?>"


def formula_to_str(f: Formula) -> str:
    if isinstance(f, Pred):
        if not f.args:
            return f.name
        return f"{f.name}(" + ", ".join(term_to_str(a) for a in f.args) + ")"
    if isinstance(f, Not):
        sub = formula_to_str(f.sub)
        if isinstance(f.sub, Pred):
            return "¬" + sub
        return "¬(" + sub + ")"
    if isinstance(f, And):
        return f"({formula_to_str(f.left)} ∧ {formula_to_str(f.right)})"
    if isinstance(f, Or):
        return f"({formula_to_str(f.left)} ∨ {formula_to_str(f.right)})"
    if isinstance(f, Implies):
        return f"({formula_to_str(f.left)} → {formula_to_str(f.right)})"
    if isinstance(f, Iff):
        return f"({formula_to_str(f.left)} ↔ {formula_to_str(f.right)})"
    if isinstance(f, ForAll):
        return f"∀{f.var}. {formula_to_str(f.body)}"
    if isinstance(f, Exists):
        return f"∃{f.var}. {formula_to_str(f.body)}"
    return "<?F>"


def literal_to_str(l: Literal) -> str:
    s = f"{l.pred}(" + ", ".join(term_to_str(a) for a in l.args) + ")"
    return s if l.positive else "¬" + s


def clause_to_str(c: Clause) -> str:
    if not c:
        return "□"
    return " ∨ ".join(sorted(literal_to_str(l) for l in c))


@dataclass
class Token:
    kind: str
    value: str


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def peek_char(self) -> str:
        if self.pos >= self.length:
            return ''
        return self.text[self.pos]

    def get_char(self) -> str:
        if self.pos >= self.length:
            return ''
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def skip_ws(self):
        while self.peek_char() and self.peek_char().isspace():
            self.get_char()

    def next_token(self) -> Token:
        self.skip_ws()
        ch = self.peek_char()
        if not ch:
            return Token("EOF", "")

        if ch == '(':
            self.get_char()
            return Token("LPAREN", "(")
        if ch == ')':
            self.get_char()
            return Token("RPAREN", ")")
        if ch == ',':
            self.get_char()
            return Token("COMMA", ",")
        if ch == '.':
            self.get_char()
            return Token("DOT", ".")

        if ch == '<':
            s3 = self.text[self.pos:self.pos+3]
            s4 = self.text[self.pos:self.pos+4]
            if s3 == "<->":
                self.pos += 3
                return Token("IFF", "<->")
            if s4 == "<=>":
                self.pos += 4
                return Token("IFF", "<=>")
        if ch == '-':
            s2 = self.text[self.pos:self.pos+2]
            if s2 == "->":
                self.pos += 2
                return Token("IMPLIES", "->")
        if ch == '=':
            s2 = self.text[self.pos:self.pos+2]
            if s2 == "=>":
                self.pos += 2
                return Token("IMPLIES", "=>")

        if ch in ['¬', '~', '!']:
            self.get_char()
            return Token("NOT", ch)
        if ch in ['∧', '&']:
            self.get_char()
            return Token("AND", ch)
        if ch in ['∨', '|']:
            self.get_char()
            return Token("OR", ch)
        if ch == '→':
            self.get_char()
            return Token("IMPLIES", "→")
        if ch == '↔':
            self.get_char()
            return Token("IFF", "↔")
        if ch in ['∀', '∃']:
            self.get_char()
            return Token("FORALL" if ch == '∀' else "EXISTS", ch)

        if ch.isalpha() or ch == '_':
            ident = []
            while self.peek_char() and (self.peek_char().isalnum() or self.peek_char() == '_'):
                ident.append(self.get_char())
            word = "".join(ident)
            lw = word.lower()
            if lw in ["forall", "all"]:
                return Token("FORALL", word)
            if lw in ["exists", "some"]:
                return Token("EXISTS", word)
            if lw in ["and", "&&"]:
                return Token("AND", word)
            if lw in ["or", "||"]:
                return Token("OR", word)
            if lw in ["not"]:
                return Token("NOT", word)
            if lw in ["implies"]:
                return Token("IMPLIES", word)
            if lw in ["iff"]:
                return Token("IFF", word)
            return Token("IDENT", word)

        raise ValueError(f"Unexpected character: {ch!r} at position {self.pos}")


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, text: str):
        self.lexer = Lexer(text)
        self.current: Token = self.lexer.next_token()
        self.bound_vars: Set[str] = set()

    def eat(self, kind: str):
        if self.current.kind == kind:
            self.current = self.lexer.next_token()
        else:
            raise ParseError(f"Expected {kind}, got {self.current.kind} ({self.current.value})")

    def parse(self) -> Formula:
        f = self.parse_iff()
        if self.current.kind != "EOF":
            raise ParseError(f"Unexpected token at end: {self.current.kind} {self.current.value}")
        return f

    def parse_iff(self) -> Formula:
        left = self.parse_implies()
        while self.current.kind == "IFF":
            self.eat("IFF")
            right = self.parse_implies()
            left = Iff(left, right)
        return left

    def parse_implies(self) -> Formula:
        left = self.parse_or()
        while self.current.kind == "IMPLIES":
            self.eat("IMPLIES")
            right = self.parse_or()
            left = Implies(left, right)
        return left

    def parse_or(self) -> Formula:
        left = self.parse_and()
        while self.current.kind == "OR":
            self.eat("OR")
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Formula:
        left = self.parse_not()
        while self.current.kind == "AND":
            self.eat("AND")
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self) -> Formula:
        if self.current.kind == "NOT":
            self.eat("NOT")
            sub = self.parse_not()
            return Not(sub)
        else:
            return self.parse_quantified_or_atom()

    def parse_quantified_or_atom(self) -> Formula:
        if self.current.kind in ("FORALL", "EXISTS"):
            return self.parse_quantifier()
        else:
            return self.parse_atom()

    def parse_quantifier(self) -> Formula:
        kind = self.current.kind
        self.eat(kind)

        vars_: List[str] = []
        if self.current.kind != "IDENT":
            raise ParseError("Expected variable name after quantifier")
        while self.current.kind == "IDENT":
            vname = self.current.value
            vars_.append(vname)
            self.bound_vars.add(vname)
            self.eat("IDENT")
            if self.current.kind == "COMMA":
                self.eat("COMMA")
            else:
                break

        if self.current.kind == "DOT":
            self.eat("DOT")

        body = self.parse_iff()
        result: Formula = body
        for v in reversed(vars_):
            if kind == "FORALL":
                result = ForAll(v, result)
            else:
                result = Exists(v, result)
        return result

    def parse_atom(self) -> Formula:
        if self.current.kind == "LPAREN":
            self.eat("LPAREN")
            f = self.parse_iff()
            if self.current.kind != "RPAREN":
                raise ParseError("Expected ')'")
            self.eat("RPAREN")
            return f

        if self.current.kind == "IDENT":
            name = self.current.value
            self.eat("IDENT")
            args: List[Term] = []
            if self.current.kind == "LPAREN":
                self.eat("LPAREN")
                if self.current.kind != "RPAREN":
                    args.append(self.parse_term())
                    while self.current.kind == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_term())
                if self.current.kind != "RPAREN":
                    raise ParseError("Expected ')' after predicate arguments")
                self.eat("RPAREN")
            return Pred(name, tuple(args))

        raise ParseError(f"Unexpected token in atom: {self.current.kind} {self.current.value}")

    def parse_term(self) -> Term:
        if self.current.kind != "IDENT":
            raise ParseError(f"Expected term, got {self.current.kind}")
        name = self.current.value
        self.eat("IDENT")

        if self.current.kind == "LPAREN":
            self.eat("LPAREN")
            args: List[Term] = []
            if self.current.kind != "RPAREN":
                args.append(self.parse_term())
                while self.current.kind == "COMMA":
                    self.eat("COMMA")
                    args.append(self.parse_term())
            if self.current.kind != "RPAREN":
                raise ParseError("Expected ')' after function arguments")
            self.eat("RPAREN")
            return Func(name, tuple(args))

        if name in self.bound_vars:
            return Var(name)
        else:
            return Const(name)


def parse_formula(text: str) -> Formula:
    return Parser(text).parse()


def eliminate_implications(f: Formula) -> Formula:
    if isinstance(f, Pred):
        return f
    if isinstance(f, Not):
        return Not(eliminate_implications(f.sub))
    if isinstance(f, And):
        return And(eliminate_implications(f.left), eliminate_implications(f.right))
    if isinstance(f, Or):
        return Or(eliminate_implications(f.left), eliminate_implications(f.right))
    if isinstance(f, Implies):
        return Or(Not(eliminate_implications(f.left)), eliminate_implications(f.right))
    if isinstance(f, Iff):
        a = eliminate_implications(f.left)
        b = eliminate_implications(f.right)
        return And(Or(Not(a), b), Or(Not(b), a))
    if isinstance(f, ForAll):
        return ForAll(f.var, eliminate_implications(f.body))
    if isinstance(f, Exists):
        return Exists(f.var, eliminate_implications(f.body))
    raise TypeError("Unknown formula in eliminate_implications")


def to_nnf(f: Formula) -> Formula:
    if isinstance(f, Pred):
        return f
    if isinstance(f, Not):
        g = f.sub
        if isinstance(g, Pred):
            return f
        if isinstance(g, Not):
            return to_nnf(g.sub)
        if isinstance(g, And):
            return Or(to_nnf(Not(g.left)), to_nnf(Not(g.right)))
        if isinstance(g, Or):
            return And(to_nnf(Not(g.left)), to_nnf(Not(g.right)))
        if isinstance(g, ForAll):
            return Exists(g.var, to_nnf(Not(g.body)))
        if isinstance(g, Exists):
            return ForAll(g.var, to_nnf(Not(g.body)))
        raise TypeError("Unexpected formula under Not in NNF")
    if isinstance(f, And):
        return And(to_nnf(f.left), to_nnf(f.right))
    if isinstance(f, Or):
        return Or(to_nnf(f.left), to_nnf(f.right))
    if isinstance(f, ForAll):
        return ForAll(f.var, to_nnf(f.body))
    if isinstance(f, Exists):
        return Exists(f.var, to_nnf(f.body))
    raise TypeError("Unknown formula in to_nnf")


def collect_var_names(f: Formula, acc: Set[str]):
    if isinstance(f, Pred):
        for t in f.args:
            collect_var_names_term(t, acc)
    elif isinstance(f, Not):
        collect_var_names(f.sub, acc)
    elif isinstance(f, (And, Or, Implies, Iff)):
        collect_var_names(f.left, acc)
        collect_var_names(f.right, acc)
    elif isinstance(f, (ForAll, Exists)):
        acc.add(f.var)
        collect_var_names(f.body, acc)


def collect_var_names_term(t: Term, acc: Set[str]):
    if isinstance(t, Var):
        acc.add(t.name)
    elif isinstance(t, Func):
        for a in t.args:
            collect_var_names_term(a, acc)


def standardize_apart(f: Formula) -> Formula:
    used: Set[str] = set()
    collect_var_names(f, used)
    counter = [0]

    def fresh_name(base: str = "V") -> str:
        counter[0] += 1
        name = f"{base}{counter[0]}"
        while name in used:
            counter[0] += 1
            name = f"{base}{counter[0]}"
        used.add(name)
        return name

    def std_term(t: Term, env: Dict[str, str]) -> Term:
        if isinstance(t, Var):
            new = env.get(t.name, t.name)
            return Var(new)
        if isinstance(t, Const):
            return t
        if isinstance(t, Func):
            return Func(t.name, tuple(std_term(a, env) for a in t.args))
        return t

    def std_fm(phi: Formula, env: Dict[str, str]) -> Formula:
        if isinstance(phi, Pred):
            return Pred(phi.name, tuple(std_term(a, env) for a in phi.args))
        if isinstance(phi, Not):
            return Not(std_fm(phi.sub, env))
        if isinstance(phi, And):
            return And(std_fm(phi.left, env), std_fm(phi.right, env))
        if isinstance(phi, Or):
            return Or(std_fm(phi.left, env), std_fm(phi.right, env))
        if isinstance(phi, Implies):
            return Implies(std_fm(phi.left, env), std_fm(phi.right, env))
        if isinstance(phi, Iff):
            return Iff(std_fm(phi.left, env), std_fm(phi.right, env))
        if isinstance(phi, ForAll):
            new_v = fresh_name(phi.var)
            new_env = env.copy()
            new_env[phi.var] = new_v
            return ForAll(new_v, std_fm(phi.body, new_env))
        if isinstance(phi, Exists):
            new_v = fresh_name(phi.var)
            new_env = env.copy()
            new_env[phi.var] = new_v
            return Exists(new_v, std_fm(phi.body, new_env))
        raise TypeError("Unknown formula in standardize_apart")

    return std_fm(f, {})


def skolemize(f: Formula) -> Formula:
    skolem_counter = [0]

    def fresh_skolem_name() -> str:
        skolem_counter[0] += 1
        return f"SK{skolem_counter[0]}"

    def replace_var_term(t: Term, var: str, repl: Term) -> Term:
        if isinstance(t, Var):
            if t.name == var:
                return repl
            return t
        if isinstance(t, Const):
            return t
        if isinstance(t, Func):
            return Func(t.name, tuple(replace_var_term(a, var, repl) for a in t.args))
        return t

    def replace_var_fm(phi: Formula, var: str, repl: Term) -> Formula:
        if isinstance(phi, Pred):
            return Pred(phi.name, tuple(replace_var_term(a, var, repl) for a in phi.args))
        if isinstance(phi, Not):
            return Not(replace_var_fm(phi.sub, var, repl))
        if isinstance(phi, And):
            return And(replace_var_fm(phi.left, var, repl),
                       replace_var_fm(phi.right, var, repl))
        if isinstance(phi, Or):
            return Or(replace_var_fm(phi.left, var, repl),
                      replace_var_fm(phi.right, var, repl))
        if isinstance(phi, ForAll):
            return ForAll(phi.var, replace_var_fm(phi.body, var, repl))
        if isinstance(phi, Exists):
            return Exists(phi.var, replace_var_fm(phi.body, var, repl))
        raise TypeError("Unknown formula in replace_var_fm")

    def sko(phi: Formula, univ_vars: List[str]) -> Formula:
        if isinstance(phi, Pred):
            return phi
        if isinstance(phi, Not):
            return Not(sko(phi.sub, univ_vars))
        if isinstance(phi, And):
            return And(sko(phi.left, univ_vars), sko(phi.right, univ_vars))
        if isinstance(phi, Or):
            return Or(sko(phi.left, univ_vars), sko(phi.right, univ_vars))
        if isinstance(phi, ForAll):
            return ForAll(phi.var, sko(phi.body, univ_vars + [phi.var]))
        if isinstance(phi, Exists):
            sk_name = fresh_skolem_name()
            if univ_vars:
                args = tuple(Var(v) for v in univ_vars)
                sk_term: Term = Func(sk_name, args)
            else:
                sk_term = Const(sk_name)
            new_body = replace_var_fm(phi.body, phi.var, sk_term)
            return sko(new_body, univ_vars)
        raise TypeError("Unknown formula in skolemize")

    return sko(f, [])


def drop_universal(f: Formula) -> Formula:
    if isinstance(f, Pred):
        return f
    if isinstance(f, Not):
        return Not(drop_universal(f.sub))
    if isinstance(f, And):
        return And(drop_universal(f.left), drop_universal(f.right))
    if isinstance(f, Or):
        return Or(drop_universal(f.left), drop_universal(f.right))
    if isinstance(f, ForAll):
        return drop_universal(f.body)
    if isinstance(f, Exists):
        return drop_universal(f.body)
    raise TypeError("Unknown formula in drop_universal")


def to_cnf_formula(f: Formula) -> Formula:
    def distribute(a: Formula, b: Formula) -> Formula:
        if isinstance(a, And):
            return And(distribute(a.left, b), distribute(a.right, b))
        if isinstance(b, And):
            return And(distribute(a, b.left), distribute(a, b.right))
        return Or(a, b)

    def cnf(phi: Formula) -> Formula:
        if isinstance(phi, Pred) or (isinstance(phi, Not) and isinstance(phi.sub, Pred)):
            return phi
        if isinstance(phi, And):
            return And(cnf(phi.left), cnf(phi.right))
        if isinstance(phi, Or):
            left = cnf(phi.left)
            right = cnf(phi.right)
            return distribute(left, right)
        raise TypeError("Formula not in proper NNF for CNF conversion")

    return cnf(f)


def is_literal_formula(phi: Formula) -> bool:
    return isinstance(phi, Pred) or (isinstance(phi, Not) and isinstance(phi.sub, Pred))


def literal_from_formula(phi: Formula) -> Literal:
    if isinstance(phi, Pred):
        return Literal(phi.name, phi.args, True)
    if isinstance(phi, Not) and isinstance(phi.sub, Pred):
        return Literal(phi.sub.name, phi.sub.args, False)
    raise ValueError("Not a literal formula")


def extract_clauses(phi: Formula) -> List[Clause]:
    def collect_clause(disj: Formula, acc: Set[Literal]):
        if isinstance(disj, Or):
            collect_clause(disj.left, acc)
            collect_clause(disj.right, acc)
        else:
            if not is_literal_formula(disj):
                raise ValueError("Non-literal in clause")
            acc.add(literal_from_formula(disj))

    if isinstance(phi, And):
        left_clauses = extract_clauses(phi.left)
        right_clauses = extract_clauses(phi.right)
        return left_clauses + right_clauses
    else:
        lits: Set[Literal] = set()
        collect_clause(phi, lits)
        return [frozenset(lits)]


def collect_vars_in_term(t: Term, acc: Set[str]):
    if isinstance(t, Var):
        acc.add(t.name)
    elif isinstance(t, Func):
        for a in t.args:
            collect_vars_in_term(a, acc)


def collect_vars_in_clause(c: Clause) -> Set[str]:
    acc: Set[str] = set()
    for lit in c:
        for t in lit.args:
            collect_vars_in_term(t, acc)
    return acc


def substitute_term_vars(t: Term, mapping: Dict[str, str]) -> Term:
    if isinstance(t, Var):
        name = mapping.get(t.name, t.name)
        return Var(name)
    if isinstance(t, Const):
        return t
    if isinstance(t, Func):
        return Func(t.name, tuple(substitute_term_vars(a, mapping) for a in t.args))
    return t


def rename_clause_vars(c: Clause, name_gen) -> Clause:
    vars_in_clause = collect_vars_in_clause(c)
    mapping: Dict[str, str] = {}
    for v in vars_in_clause:
        mapping[v] = next(name_gen)
    new_lits = set()
    for lit in c:
        new_args = tuple(substitute_term_vars(t, mapping) for t in lit.args)
        new_lits.add(Literal(lit.pred, new_args, lit.positive))
    return frozenset(new_lits)


def clause_var_name_generator():
    i = 0
    while True:
        i += 1
        yield f"V{i}"


def standardize_clauses_apart(clauses: List[Clause]) -> List[Clause]:
    gen = clause_var_name_generator()
    new_clauses: List[Clause] = []
    for c in clauses:
        new_clauses.append(rename_clause_vars(c, gen))
    return new_clauses


Substitution = Dict[str, Term]


def apply_subst_term(t: Term, subst: Substitution) -> Term:
    if isinstance(t, Var):
        if t.name in subst:
            return apply_subst_term(subst[t.name], subst)
        return t
    if isinstance(t, Const):
        return t
    if isinstance(t, Func):
        return Func(t.name, tuple(apply_subst_term(a, subst) for a in t.args))
    return t


def apply_subst_literal(lit: Literal, subst: Substitution) -> Literal:
    new_args = tuple(apply_subst_term(a, subst) for a in lit.args)
    return Literal(lit.pred, new_args, lit.positive)


def apply_subst_clause(c: Clause, subst: Substitution) -> Clause:
    return frozenset(apply_subst_literal(l, subst) for l in c)


def occurs_in(var: str, t: Term, subst: Substitution) -> bool:
    t = apply_subst_term(t, subst)
    if isinstance(t, Var):
        return t.name == var
    if isinstance(t, Func):
        return any(occurs_in(var, a, subst) for a in t.args)
    return False


def unify_terms(t1: Term, t2: Term, subst: Optional[Substitution] = None) -> Optional[Substitution]:
    if subst is None:
        subst = {}
    t1 = apply_subst_term(t1, subst)
    t2 = apply_subst_term(t2, subst)

    if isinstance(t1, Var):
        if isinstance(t2, Var) and t1.name == t2.name:
            return subst
        if occurs_in(t1.name, t2, subst):
            return None
        subst = dict(subst)
        subst[t1.name] = t2
        return subst

    if isinstance(t2, Var):
        return unify_terms(t2, t1, subst)

    if isinstance(t1, Const) and isinstance(t2, Const):
        if t1.name == t2.name:
            return subst
        return None

    if isinstance(t1, Func) and isinstance(t2, Func):
        if t1.name != t2.name or len(t1.args) != len(t2.args):
            return None
        for a, b in zip(t1.args, t2.args):
            subst = unify_terms(a, b, subst)
            if subst is None:
                return None
        return subst

    return None


def unify_literals(l1: Literal, l2: Literal) -> Optional[Substitution]:
    if l1.pred != l2.pred:
        return None
    if len(l1.args) != len(l2.args):
        return None
    subst: Substitution = {}
    for a, b in zip(l1.args, l2.args):
        subst = unify_terms(a, b, subst)
        if subst is None:
            return None
    return subst


def resolve_clauses(c1: Clause, c2: Clause):
    resolvents = []
    for l1 in c1:
        for l2 in c2:
            if l1.pred == l2.pred and l1.positive != l2.positive:
                subst = unify_literals(l1, l2)
                if subst is None:
                    continue
                new_lits = set()
                for lit in c1:
                    if lit is not l1:
                        new_lits.add(apply_subst_literal(lit, subst))
                for lit in c2:
                    if lit is not l2:
                        new_lits.add(apply_subst_literal(lit, subst))

                skip_clause = False
                for lit in list(new_lits):
                    if Literal(lit.pred, lit.args, not lit.positive) in new_lits:
                        skip_clause = True
                        break
                if skip_clause:
                    continue

                resolvents.append((frozenset(new_lits), subst, l1, l2))
    return resolvents


def resolution(clauses: Iterable[Clause], max_steps: int = 1000):
    clauses = list(clauses)
    clauses = standardize_clauses_apart(clauses)

    all_clauses = set(clauses)
    processed: List[Clause] = []
    agenda: List[Clause] = list(clauses)
    steps: List[str] = []
    step_count = 0

    while agenda and step_count < max_steps:
        C = agenda.pop(0)

        for D in processed:
            res_list = resolve_clauses(C, D)
            for R, subst, l1, l2 in res_list:
                step_count += 1

                subst_str = ", ".join(f"{v}={term_to_str(t)}" for v, t in subst.items()) or "∅"
                step_str = (
                    f"Step {step_count}: Resolve [{clause_to_str(C)}] and "
                    f"[{clause_to_str(D)}] on {literal_to_str(l1)} / {literal_to_str(l2)} "
                    f"with σ = {{{subst_str}}} to get [{clause_to_str(R)}]."
                )
                steps.append(step_str)

                if not R:
                    return True, steps

                if R not in all_clauses:
                    all_clauses.add(R)
                    agenda.append(R)

                if step_count >= max_steps:
                    break
            if step_count >= max_steps:
                break

        processed.append(C)

    return False, steps


def prove_formula(formula_str: str, max_steps: int = 1000) -> Dict[str, Any]:
    steps: List[str] = []

    try:
        parsed = parse_formula(formula_str)
    except ParseError as e:
        return {
            "is_valid": None,
            "message": f"Parse error: {e}",
            "steps": []
        }
    except Exception as e:
        return {
            "is_valid": None,
            "message": f"Unexpected error while parsing: {e}",
            "steps": []
        }

    steps.append("Input formula F:")
    steps.append("  " + formula_to_str(parsed))

    neg = Not(parsed)
    steps.append("Negate formula (we prove validity of F by refuting ¬F):")
    steps.append("  ¬F = " + formula_to_str(neg))

    noimpl = eliminate_implications(neg)
    steps.append("1. Eliminate →, ↔ from ¬F:")
    steps.append("  " + formula_to_str(noimpl))

    nnf = to_nnf(noimpl)
    steps.append("2. Convert to negation normal form (NNF):")
    steps.append("  " + formula_to_str(nnf))

    std = standardize_apart(nnf)
    steps.append("3. Standardize variables apart:")
    steps.append("  " + formula_to_str(std))

    skol = skolemize(std)
    steps.append("4. Skolemize (remove ∃ using Skolem functions/constants):")
    steps.append("  " + formula_to_str(skol))

    dropped = drop_universal(skol)
    steps.append("5. Drop universal quantifiers (all variables are now implicitly ∀):")
    steps.append("  " + formula_to_str(dropped))

    try:
        cnf_formula = to_cnf_formula(dropped)
    except Exception as e:
        return {
            "is_valid": None,
            "message": f"Error converting to CNF: {e}",
            "steps": steps
        }
    steps.append("6. Convert matrix to CNF:")
    steps.append("  " + formula_to_str(cnf_formula))

    try:
        clauses = extract_clauses(cnf_formula)
    except Exception as e:
        return {
            "is_valid": None,
            "message": f"Error extracting clauses: {e}",
            "steps": steps
        }

    steps.append("7. Clause set for ¬F:")
    for i, c in enumerate(clauses, 1):
        steps.append(f"  C{i}: {clause_to_str(c)}")

    is_unsat, res_steps = resolution(clauses, max_steps=max_steps)
    steps.append("8. Resolution steps:")
    if res_steps:
        steps.extend("  " + s for s in res_steps)
    else:
        steps.append("  (No resolution steps performed.)")

    if is_unsat:
        msg = "VALID: derived empty clause, so ¬F is unsatisfiable and F is valid."
        is_valid_flag = True
    else:
        msg = (
            "NOT PROVED VALID: did not derive empty clause from ¬F within the step limit. "
            "F may be satisfiable or the search space is too large."
        )
        is_valid_flag = False

    return {
        "is_valid": is_valid_flag,
        "message": msg,
        "steps": steps
    }


if __name__ == "__main__":
    examples = [
        "forall x. P(x) -> P(x)",
        "(forall x. (P(x) -> Q(x)) and forall x. (Q(x) -> R(x))) -> forall x. (P(x) -> R(x))"
    ]
    for s in examples:
        print("Formula:", s)
        r = prove_formula(s, max_steps=2000)
        print("  is_valid:", r["is_valid"])
        print("  message:", r["message"])
        print("  steps:", len(r["steps"]), "lines")
        print()
