class Term:
    pass

class Var(Term):
    def __init__(self, name):
        self.name = name


class Const(Term):
    def __init__(self, name):
        self.name = name


class Func(Term):
    def __init__(self, name, args):
        self.name = name
        self.args = tuple(args)


class Formula:
    pass


class Pred(Formula):
    def __init__(self, name, args):
        self.name = name
        self.args = tuple(args)


class Not(Formula):
    def __init__(self, sub):
        self.sub = sub


class And(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Or(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Implies(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Iff(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class ForAll(Formula):
    def __init__(self, var, body):
        self.var = var
        self.body = body


class Exists(Formula):
    def __init__(self, var, body):
        self.var = var
        self.body = body


class Literal:
    def __init__(self, pred, args, positive=True):
        self.pred = pred
        self.args = tuple(args)
        self.positive = positive

    def __eq__(self, o):
        return (
            isinstance(o, Literal)
            and (self.pred, self.args, self.positive) == (o.pred, o.args, o.positive)
        )

    def __hash__(self):
        return hash((self.pred, self.args, self.positive))


def term_to_str(t):
    if isinstance(t, Var) or isinstance(t, Const):
        return t.name
    if isinstance(t, Func):
        if not t.args:
            return t.name
        return "%s(%s)" % (t.name, ", ".join(term_to_str(a) for a in t.args))
    return "<?>"



def formula_to_str(f):
    if isinstance(f, Pred):
        if not f.args:
            return f.name
        return "%s(%s)" % (f.name, ", ".join(term_to_str(a) for a in f.args))
    if isinstance(f, Not):
        s = formula_to_str(f.sub)
        if isinstance(f.sub, Pred):
            return "¬" + s
        return "¬(" + s + ")"
    if isinstance(f, And):
        return "(%s ∧ %s)" % (formula_to_str(f.left), formula_to_str(f.right))
    if isinstance(f, Or):
        return "(%s ∨ %s)" % (formula_to_str(f.left), formula_to_str(f.right))
    if isinstance(f, Implies):
        return "(%s → %s)" % (formula_to_str(f.left), formula_to_str(f.right))
    if isinstance(f, Iff):
        return "(%s ↔ %s)" % (formula_to_str(f.left), formula_to_str(f.right))
    if isinstance(f, ForAll):
        return "∀%s. %s" % (f.var, formula_to_str(f.body))
    if isinstance(f, Exists):
        return "∃%s. %s" % (f.var, formula_to_str(f.body))
    return "<?F>"


def literal_to_str(l):
    s = "%s(%s)" % (l.pred, ", ".join(term_to_str(a) for a in l.args))
    return s if l.positive else "¬" + s


def clause_to_str(c):
    if not c:
        return "□"
    return " ∨ ".join(sorted(literal_to_str(l) for l in c))


class Token:
    def __init__(self, kind, value):
        self.kind = kind
        self.value = value


class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def peek_char(self):
        if self.pos >= self.length:
            return ""
        return self.text[self.pos]

    def get_char(self):
        if self.pos >= self.length:
            return ""
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def skip_ws(self):
        while self.peek_char() and self.peek_char().isspace():
            self.get_char()

    def next_token(self):
        self.skip_ws()
        ch = self.peek_char()
        if not ch:
            return Token("EOF", "")
        if ch == "(":
            self.get_char()
            return Token("LPAREN", "(")
        if ch == ")":
            self.get_char()
            return Token("RPAREN", ")")
        if ch == ",":
            self.get_char()
            return Token("COMMA", ",")
        if ch == ".":
            self.get_char()
            return Token("DOT", ".")
        if ch == "<":
            s3 = self.text[self.pos:self.pos + 3]
            s4 = self.text[self.pos:self.pos + 4]
            if s3 == "<->":
                self.pos += 3
                return Token("IFF", "<->")
            if s4 == "<=>":
                self.pos += 4
                return Token("IFF", "<=>")
        if ch == "-":
            s2 = self.text[self.pos:self.pos + 2]
            if s2 == "->":
                self.pos += 2
                return Token("IMPLIES", "->")
        if ch == "=":
            s2 = self.text[self.pos:self.pos + 2]
            if s2 == "=>":
                self.pos += 2
                return Token("IMPLIES", "=>")
        if ch in ("¬", "~", "!"):
            self.get_char()
            return Token("NOT", ch)
        if ch in ("∧", "&"):
            self.get_char()
            return Token("AND", ch)
        if ch in ("∨", "|"):
            self.get_char()
            return Token("OR", ch)
        if ch == "→":
            self.get_char()
            return Token("IMPLIES", "→")
        if ch == "↔":
            self.get_char()
            return Token("IFF", "↔")
        if ch in ("∀", "∃"):
            self.get_char()
            return Token("FORALL" if ch == "∀" else "EXISTS", ch)
        if ch.isalpha() or ch == "_":
            ident = []
            while self.peek_char() and (self.peek_char().isalnum() or self.peek_char() == "_"):
                ident.append(self.get_char())
            word = "".join(ident)
            kw = {
                "forall": "FORALL",
                "all": "FORALL",
                "exists": "EXISTS",
                "some": "EXISTS",
                "and": "AND",
                "or": "OR",
                "not": "NOT",
                "iff": "IFF",
                "implies": "IMPLIES",
            }
            kind = kw.get(word.lower())
            if kind:
                return Token(kind, word)
            return Token("IDENT", word)
        raise ValueError("Unexpected character %r at %d" % (ch, self.pos))


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, text):
        self.lexer = Lexer(text)
        self.current = self.lexer.next_token()
        self.bound_vars = set()

    def eat(self, kind):
        if self.current.kind == kind:
            self.current = self.lexer.next_token()
        else:
            raise ParseError("Expected %s, got %s (%s)" % (kind, self.current.kind, self.current.value))

    def parse(self):
        f = self.parse_iff()
        if self.current.kind != "EOF":
            raise ParseError("Unexpected token at end: %s %s" % (self.current.kind, self.current.value))
        return f

    def parse_iff(self):
        left = self.parse_implies()
        while self.current.kind == "IFF":
            self.eat("IFF")
            right = self.parse_implies()
            left = Iff(left, right)
        return left

    def parse_implies(self):
        left = self.parse_or()
        while self.current.kind == "IMPLIES":
            self.eat("IMPLIES")
            right = self.parse_or()
            left = Implies(left, right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.current.kind == "OR":
            self.eat("OR")
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.current.kind == "AND":
            self.eat("AND")
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self):
        if self.current.kind == "NOT":
            self.eat("NOT")
            return Not(self.parse_not())
        return self.parse_quantified_or_atom()

    def parse_quantified_or_atom(self):
        if self.current.kind in ("FORALL", "EXISTS"):
            return self.parse_quantifier()
        return self.parse_atom()

    def parse_quantifier(self):
        kind = self.current.kind
        self.eat(kind)
        vars_ = []
        if self.current.kind != "IDENT":
            raise ParseError("Expected variable name after quantifier")
        while self.current.kind == "IDENT":
            v = self.current.value
            vars_.append(v)
            self.bound_vars.add(v)
            self.eat("IDENT")
            if self.current.kind == "COMMA":
                self.eat("COMMA")
            else:
                break
        if self.current.kind == "DOT":
            self.eat("DOT")
        body = self.parse_iff()
        r = body
        for v in reversed(vars_):
            r = ForAll(v, r) if kind == "FORALL" else Exists(v, r)
        return r

    def parse_atom(self):
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
            args = []
            if self.current.kind == "LPAREN":
                self.eat("LPAREN")
                if self.current.kind != "RPAREN":
                    args.append(self.parse_term())
                    while self.current.kind == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_term())
                if self.current.kind != "RPAREN":
                    raise ParseError("Expected ')' after predicate args")
                self.eat("RPAREN")
            return Pred(name, tuple(args))
        raise ParseError("Unexpected token in atom: %s %s" % (self.current.kind, self.current.value))

    def parse_term(self):
        if self.current.kind != "IDENT":
            raise ParseError("Expected term, got %s" % self.current.kind)
        name = self.current.value
        self.eat("IDENT")
        if self.current.kind == "LPAREN":
            self.eat("LPAREN")
            args = []
            if self.current.kind != "RPAREN":
                args.append(self.parse_term())
                while self.current.kind == "COMMA":
                    self.eat("COMMA")
                    args.append(self.parse_term())
            if self.current.kind != "RPAREN":
                raise ParseError("Expected ')' after function args")
            self.eat("RPAREN")
            return Func(name, tuple(args))
        return Var(name) if name in self.bound_vars else Const(name)


def parse_formula(text):
    return Parser(text).parse()


def eliminate_implications(f):
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
    raise TypeError("bad formula in eliminate_implications")


def to_nnf(f):
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
        raise TypeError("bad formula under Not")
    if isinstance(f, And):
        return And(to_nnf(f.left), to_nnf(f.right))
    if isinstance(f, Or):
        return Or(to_nnf(f.left), to_nnf(f.right))
    if isinstance(f, ForAll):
        return ForAll(f.var, to_nnf(f.body))
    if isinstance(f, Exists):
        return Exists(f.var, to_nnf(f.body))
    raise TypeError("bad formula in to_nnf")


def vars_in_term(t, acc):
    if isinstance(t, Var):
        acc.add(t.name)
    elif isinstance(t, Func):
        for a in t.args:
            vars_in_term(a, acc)


def collect_var_names(f, acc):
    if isinstance(f, Pred):
        for t in f.args:
            vars_in_term(t, acc)
    elif isinstance(f, Not):
        collect_var_names(f.sub, acc)
    elif isinstance(f, (And, Or, Implies, Iff)):
        collect_var_names(f.left, acc)
        collect_var_names(f.right, acc)
    elif isinstance(f, (ForAll, Exists)):
        acc.add(f.var)
        collect_var_names(f.body, acc)


def standardize_apart(f):
    used = set()
    collect_var_names(f, used)
    counter = [0]

    def fresh(base="V"):
        counter[0] += 1
        name = "%s%d" % (base, counter[0])
        while name in used:
            counter[0] += 1
            name = "%s%d" % (base, counter[0])
        used.add(name)
        return name

    def std_term(t, env):
        if isinstance(t, Var):
            return Var(env.get(t.name, t.name))
        if isinstance(t, Const):
            return t
        if isinstance(t, Func):
            return Func(t.name, tuple(std_term(a, env) for a in t.args))
        return t

    def std_fm(phi, env):
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
            nv = fresh(phi.var)
            e = dict(env)
            e[phi.var] = nv
            return ForAll(nv, std_fm(phi.body, e))
        if isinstance(phi, Exists):
            nv = fresh(phi.var)
            e = dict(env)
            e[phi.var] = nv
            return Exists(nv, std_fm(phi.body, e))
        raise TypeError("bad formula in standardize_apart")

    return std_fm(f, {})


def skolemize(f):
    skc = [0]

    def fresh():
        skc[0] += 1
        return "SK%d" % skc[0]

    def repl_term(t, var, repl):
        if isinstance(t, Var):
            return repl if t.name == var else t
        if isinstance(t, Const):
            return t
        if isinstance(t, Func):
            return Func(t.name, tuple(repl_term(a, var, repl) for a in t.args))
        return t

    def repl_fm(phi, var, repl):
        if isinstance(phi, Pred):
            return Pred(phi.name, tuple(repl_term(a, var, repl) for a in phi.args))
        if isinstance(phi, Not):
            return Not(repl_fm(phi.sub, var, repl))
        if isinstance(phi, And):
            return And(repl_fm(phi.left, var, repl), repl_fm(phi.right, var, repl))
        if isinstance(phi, Or):
            return Or(repl_fm(phi.left, var, repl), repl_fm(phi.right, var, repl))
        if isinstance(phi, ForAll):
            return ForAll(phi.var, repl_fm(phi.body, var, repl))
        if isinstance(phi, Exists):
            return Exists(phi.var, repl_fm(phi.body, var, repl))
        raise TypeError("bad formula in repl_fm")

    def sko(phi, univ):
        if isinstance(phi, Pred):
            return phi
        if isinstance(phi, Not):
            return Not(sko(phi.sub, univ))
        if isinstance(phi, And):
            return And(sko(phi.left, univ), sko(phi.right, univ))
        if isinstance(phi, Or):
            return Or(sko(phi.left, univ), sko(phi.right, univ))
        if isinstance(phi, ForAll):
            return ForAll(phi.var, sko(phi.body, univ + [phi.var]))
        if isinstance(phi, Exists):
            name = fresh()
            if univ:
                sk = Func(name, tuple(Var(v) for v in univ))
            else:
                sk = Const(name)
            return sko(repl_fm(phi.body, phi.var, sk), univ)
        raise TypeError("bad formula in skolemize")

    return sko(f, [])


def drop_universal(f):
    if isinstance(f, Pred):
        return f
    if isinstance(f, Not):
        return Not(drop_universal(f.sub))
    if isinstance(f, And):
        return And(drop_universal(f.left), drop_universal(f.right))
    if isinstance(f, Or):
        return Or(drop_universal(f.left), drop_universal(f.right))
    if isinstance(f, (ForAll, Exists)):
        return drop_universal(f.body)
    raise TypeError("bad formula in drop_universal")


def to_cnf_formula(f):
    def dist(a, b):
        if isinstance(a, And):
            return And(dist(a.left, b), dist(a.right, b))
        if isinstance(b, And):
            return And(dist(a, b.left), dist(a, b.right))
        return Or(a, b)

    def cnf(phi):
        if isinstance(phi, Pred) or (isinstance(phi, Not) and isinstance(phi.sub, Pred)):
            return phi
        if isinstance(phi, And):
            return And(cnf(phi.left), cnf(phi.right))
        if isinstance(phi, Or):
            return dist(cnf(phi.left), cnf(phi.right))
        raise TypeError("bad formula for cnf")

    return cnf(f)


def is_literal_formula(phi):
    return isinstance(phi, Pred) or (isinstance(phi, Not) and isinstance(phi.sub, Pred))


def literal_from_formula(phi):
    if isinstance(phi, Pred):
        return Literal(phi.name, phi.args, True)
    if isinstance(phi, Not) and isinstance(phi.sub, Pred):
        return Literal(phi.sub.name, phi.sub.args, False)
    raise ValueError("not literal")


def extract_clauses(phi):
    def collect(disj, acc):
        if isinstance(disj, Or):
            collect(disj.left, acc)
            collect(disj.right, acc)
        else:
            if not is_literal_formula(disj):
                raise ValueError("Non-literal in clause")
            acc.add(literal_from_formula(disj))

    if isinstance(phi, And):
        return extract_clauses(phi.left) + extract_clauses(phi.right)
    lits = set()
    collect(phi, lits)
    return [frozenset(lits)]


def vars_in_clause(c):
    s = set()
    for lit in c:
        for t in lit.args:
            vars_in_term(t, s)
    return s


def substitute_term_vars(t, mapping):
    if isinstance(t, Var):
        return Var(mapping.get(t.name, t.name))
    if isinstance(t, Const):
        return t
    if isinstance(t, Func):
        return Func(t.name, tuple(substitute_term_vars(a, mapping) for a in t.args))
    return t


def rename_clause_vars(c, name_gen):
    vs = vars_in_clause(c)
    mapping = {v: next(name_gen) for v in vs}
    new = set()
    for lit in c:
        new_args = tuple(substitute_term_vars(t, mapping) for t in lit.args)
        new.add(Literal(lit.pred, new_args, lit.positive))
    return frozenset(new)


def clause_var_name_generator():
    i = 0
    while True:
        i += 1
        yield "V%d" % i


def standardize_clauses_apart(clauses):
    gen = clause_var_name_generator()
    return [rename_clause_vars(c, gen) for c in clauses]


def apply_subst_term(t, subst):
    if isinstance(t, Var):
        if t.name in subst:
            return apply_subst_term(subst[t.name], subst)
        return t
    if isinstance(t, Const):
        return t
    if isinstance(t, Func):
        return Func(t.name, tuple(apply_subst_term(a, subst) for a in t.args))
    return t


def apply_subst_literal(lit, subst):
    return Literal(lit.pred, tuple(apply_subst_term(a, subst) for a in lit.args), lit.positive)


def occurs_in(var, t, subst):
    t = apply_subst_term(t, subst)
    if isinstance(t, Var):
        return t.name == var
    if isinstance(t, Func):
        return any(occurs_in(var, a, subst) for a in t.args)
    return False


def unify_terms(t1, t2, subst=None):
    if subst is None:
        subst = {}
    t1 = apply_subst_term(t1, subst)
    t2 = apply_subst_term(t2, subst)
    if isinstance(t1, Var):
        if isinstance(t2, Var) and t1.name == t2.name:
            return subst
        if occurs_in(t1.name, t2, subst):
            return None
        s = dict(subst)
        s[t1.name] = t2
        return s
    if isinstance(t2, Var):
        return unify_terms(t2, t1, subst)
    if isinstance(t1, Const) and isinstance(t2, Const):
        return subst if t1.name == t2.name else None
    if isinstance(t1, Func) and isinstance(t2, Func):
        if t1.name != t2.name or len(t1.args) != len(t2.args):
            return None
        s = dict(subst)
        for a, b in zip(t1.args, t2.args):
            s = unify_terms(a, b, s)
            if s is None:
                return None
        return s
    return None


def unify_literals(l1, l2):
    if l1.pred != l2.pred or len(l1.args) != len(l2.args):
        return None
    s = {}
    for a, b in zip(l1.args, l2.args):
        s = unify_terms(a, b, s)
        if s is None:
            return None
    return s


def resolve_clauses(c1, c2):
    res = []
    for l1 in c1:
        for l2 in c2:
            if l1.pred == l2.pred and l1.positive != l2.positive:
                s = unify_literals(l1, l2)
                if s is None:
                    continue
                new = set()
                for lit in c1:
                    if lit is not l1:
                        new.add(apply_subst_literal(lit, s))
                for lit in c2:
                    if lit is not l2:
                        new.add(apply_subst_literal(lit, s))
                skip = False
                for lit in list(new):
                    if Literal(lit.pred, lit.args, not lit.positive) in new:
                        skip = True
                        break
                if not skip:
                    res.append((frozenset(new), s, l1, l2))
    return res


def resolution(clauses, max_steps=1000):
    clauses = list(clauses)
    clauses = standardize_clauses_apart(clauses)
    all_clauses = set(clauses)
    processed = []
    agenda = list(clauses)
    steps = []
    n = 0
    while agenda and n < max_steps:
        C = agenda.pop(0)
        for D in processed:
            for R, s, l1, l2 in resolve_clauses(C, D):
                n += 1
                if s:
                    subst_str = ", ".join("%s=%s" % (v, term_to_str(t)) for v, t in s.items())
                else:
                    subst_str = "∅"
                steps.append(
                    "Step %d: resolve [%s] and [%s] on %s / %s with σ={%s} -> [%s]"
                    % (
                        n,
                        clause_to_str(C),
                        clause_to_str(D),
                        literal_to_str(l1),
                        literal_to_str(l2),
                        subst_str,
                        clause_to_str(R),
                    )
                )
                if not R:
                    return True, steps
                if R not in all_clauses:
                    all_clauses.add(R)
                    agenda.append(R)
                if n >= max_steps:
                    break
            if n >= max_steps:
                break
        processed.append(C)
    return False, steps


def prove_formula(formula_str, max_steps=1000):
    steps = []
    try:
        parsed = parse_formula(formula_str)
    except ParseError as e:
        return {"is_valid": None, "message": "Parse error: %s" % e, "steps": []}
    except Exception as e:
        return {"is_valid": None, "message": "Unexpected error while parsing: %s" % e, "steps": []}
    steps.append("Input formula F:")
    steps.append("  " + formula_to_str(parsed))
    neg = Not(parsed)
    steps.append("Negate formula (prove F by refuting ¬F):")
    steps.append("  ¬F = " + formula_to_str(neg))
    noimpl = eliminate_implications(neg)
    steps.append("1. Eliminate →, ↔ from ¬F:")
    steps.append("  " + formula_to_str(noimpl))
    nnf = to_nnf(noimpl)
    steps.append("2. Convert to NNF:")
    steps.append("  " + formula_to_str(nnf))
    std = standardize_apart(nnf)
    steps.append("3. Standardize variables apart:")
    steps.append("  " + formula_to_str(std))
    skol = skolemize(std)
    steps.append("4. Skolemize (remove ∃):")
    steps.append("  " + formula_to_str(skol))
    dropped = drop_universal(skol)
    steps.append("5. Drop universal quantifiers:")
    steps.append("  " + formula_to_str(dropped))
    try:
        cnf_formula = to_cnf_formula(dropped)
    except Exception as e:
        return {"is_valid": None, "message": "Error converting to CNF: %s" % e, "steps": steps}
    steps.append("6. Convert matrix to CNF:")
    steps.append("  " + formula_to_str(cnf_formula))
    try:
        clauses = extract_clauses(cnf_formula)
    except Exception as e:
        return {"is_valid": None, "message": "Error extracting clauses: %s" % e, "steps": steps}
    steps.append("7. Clause set for ¬F:")
    for i, c in enumerate(clauses, 1):
        steps.append("  C%d: %s" % (i, clause_to_str(c)))
    is_unsat, res_steps = resolution(clauses, max_steps=max_steps)
    steps.append("8. Resolution steps:")
    if res_steps:
        for s in res_steps:
            steps.append("  " + s)
    else:
        steps.append("  (no resolution steps performed)")
    if is_unsat:
        msg = "VALID: derived empty clause, so ¬F is unsatisfiable and F is valid."
        flag = True
    else:
        msg = "NOT PROVED VALID: didn't derive empty clause within step limit."
        flag = False
    return {"is_valid": flag, "message": msg, "steps": steps}
