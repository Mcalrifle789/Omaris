#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omaris — reference interpreter, written from scratch (stdlib only).

Omaris is a general-purpose language whose central idea is that certain
instructions act as storage containers in their own right, so a project's
code and data never need to live on local disk, flash drives, or external
hard drives.  It is intended for building AI models (LLMs, image, code,
chat, audio/music models) with a native model lifecycle and native
calculus operators (integrals and derivatives).

Usage:
    python omaris.py program.omr
    python omaris.py                # interactive REPL
"""

import math
import os
import struct
import wave
import webbrowser
import re
import sys
import urllib.parse

VERSION = "0.1.0"


class OmarisError(Exception):
    """A user-facing error in an Omaris program."""


class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


# ---------------------------------------------------------------------------
# Expression tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'''
    (?P<ws>\s+)
  | (?P<num>\d+\.\d+|\d+)
  | (?P<str>"(?:\\.|[^"\\])*")
  | (?P<uop>\u222b\u222b\u222b|\u222b\u222b|\u222b|\u1e8f|\u00ff)
  | (?P<op>\*\*|==|!=|<=|>=|->|[+\-*/%()<>\[\]{},:.])
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
''', re.VERBOSE)


def tokenize(text):
    toks = []
    covered = 0
    for m in _TOKEN_RE.finditer(text):
        covered += len(m.group())
        if m.lastgroup == 'ws':
            continue
        toks.append((m.lastgroup, m.group()))
    if covered != len(text):
        bad = next((ch for ch in text if not ch.isspace()
                    and not _TOKEN_RE.match(ch)), '?')
        raise OmarisError("unexpected character %r in expression: %s" % (bad, text))
    return toks


# ---------------------------------------------------------------------------
# Expression parser (recursive descent, precedence climbing)
#
# AST node shapes:
#   ('num', float|int) ('str', s) ('list', [..]) ('dict', [(k, v), ..])
#   ('var', name) ('call', name, [args]) ('idx', obj, idx)
#   ('bin', op, l, r) ('and', l, r) ('or', l, r) ('not', x) ('neg', x)
#   ('uop', symbol, x)      # integral / derivative prefix operators
# ---------------------------------------------------------------------------

class ExprParser:
    def __init__(self, text):
        self.toks = tokenize(text)
        self.i = 0
        self.src = text

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else ('eof', '')

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def at(self, val):
        return self.peek()[1] == val

    def expect(self, val):
        t = self.next()
        if t[1] != val:
            raise OmarisError("expected %r but found %r in: %s" % (val, t[1], self.src))
        return t

    def parse(self):
        node = self.parse_or()
        if self.peek()[0] != 'eof':
            raise OmarisError("unexpected %r after expression: %s" % (self.peek()[1], self.src))
        return node

    def parse_or(self):
        node = self.parse_and()
        while self.peek()[1] == 'or':
            self.next()
            node = ('or', node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.peek()[1] == 'and':
            self.next()
            node = ('and', node, self.parse_not())
        return node

    def parse_not(self):
        if self.peek()[1] == 'not':
            self.next()
            return ('not', self.parse_not())
        return self.parse_cmp()

    def parse_cmp(self):
        node = self.parse_add()
        if self.peek()[1] in ('==', '!=', '<', '<=', '>', '>='):
            op = self.next()[1]
            node = ('bin', op, node, self.parse_add())
        return node

    def parse_add(self):
        node = self.parse_mul()
        while self.peek()[1] in ('+', '-'):
            op = self.next()[1]
            node = ('bin', op, node, self.parse_mul())
        return node

    def parse_mul(self):
        node = self.parse_unary()
        while self.peek()[1] in ('*', '/', '%'):
            op = self.next()[1]
            node = ('bin', op, node, self.parse_unary())
        return node

    def parse_unary(self):
        t = self.peek()
        if t[1] == '-':
            self.next()
            return ('neg', self.parse_unary())
        if t[0] == 'uop':
            self.next()
            return ('uop', t[1], self.parse_unary())
        return self.parse_pow()

    def parse_pow(self):
        node = self.parse_postfix()
        if self.peek()[1] == '**':
            self.next()
            node = ('bin', '**', node, self.parse_unary())
        return node

    def parse_postfix(self):
        node = self.parse_atom()
        while True:
            if self.at('('):
                self.next()
                args = []
                if not self.at(')'):
                    args.append(self.parse_or())
                    while self.at(','):
                        self.next()
                        args.append(self.parse_or())
                self.expect(')')
                name = node[1] if node[0] == 'var' else None
                if name is None:
                    raise OmarisError("only named function calls are supported, in: %s" % self.src)
                node = ('call', name, args)
            elif self.at('['):
                self.next()
                idx = self.parse_or()
                self.expect(']')
                node = ('idx', node, idx)
            else:
                break
        return node

    def parse_atom(self):
        kind, val = self.next()
        if kind == 'num':
            return ('num', float(val) if '.' in val else int(val))
        if kind == 'str':
            return ('str', val[1:-1].replace('\\"', '"').replace("\\n", "\n")
                    .replace("\\t", "\t").replace("\\\\", "\\"))
        if kind == 'name':
            name = val
            while self.at('.'):
                self.next()
                nxt = self.next()
                if nxt[0] != 'name':
                    raise OmarisError("expected a name after '.' in: %s" % self.src)
                name += '.' + nxt[1]
            return ('var', name)
        if val == '(':
            node = self.parse_or()
            self.expect(')')
            return node
        if val == '[':
            items = []
            if not self.at(']'):
                items.append(self.parse_or())
                while self.at(','):
                    self.next()
                    items.append(self.parse_or())
            self.expect(']')
            return ('list', items)
        if val == '{':
            pairs = []
            if not self.at('}'):
                while True:
                    k = self.next()
                    if k[0] not in ('name', 'str'):
                        raise OmarisError("bad dict key %r in: %s" % (k[1], self.src))
                    self.expect(':')
                    pairs.append((k[1] if k[0] == 'name' else k[1], self.parse_or()))
                    if self.at(','):
                        self.next()
                        continue
                    break
            self.expect('}')
            return ('dict', pairs)
        raise OmarisError("unexpected %r in expression: %s" % (val, self.src))


# ---------------------------------------------------------------------------
# Statement classification (Omaris is line-based; every block ends with `end`)
# ---------------------------------------------------------------------------

R_LET = re.compile(r'^let\s+([A-Za-z_]\w*)\s*=\s*(.+)$')
R_OUT = re.compile(r'^out\s+(.+)$')
R_GIVE = re.compile(r'^give(?:\s+(.+))?$', re.S)
R_IF = re.compile(r'^if\s+(.+?)\s+then$')
R_ELSE = re.compile(r'^else$')
R_END = re.compile(r'^end$')
R_REPEAT = re.compile(r'^repeat\s+(.+?)(?:\s+as\s+([A-Za-z_]\w*))?$')
R_MODEL = re.compile(r'^model:create\s+([A-Za-z_]\w*)$')
R_DEF = re.compile(r'^def\.fun\.+:?\(([^)]*)\)$')
R_TR = re.compile(r'^tr\.fun\.+:?\((dist\.|dist\.;\s*value)\)\s*(.+)$')
R_STORE_MODE = re.compile(r'^store\.fun\.+:?\((open|closed)\)$')
R_STORE_START = re.compile(
    r'^store\.fun\.;\s*start'
    r'(?:\s+([A-Za-z_]\w*))?'
    r'(?:\s*/capacity:\s*(GB|TB)\s*\(\s*(\d+)\s*\)|\s*/capacity:\s*(infinite))?'
    r'(?:\s+([A-Za-z_]\w*))?$', re.I)
R_STATUS = re.compile(r'^store\.fun\.;\s*status\s+([A-Za-z_]\w*)$')
R_RUN = re.compile(r'^store\.fun\.;\s*run\s+([A-Za-z_]\w*)$')
R_POCKET_ADD = re.compile(r'^pocket:add\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)$')
R_POCKET_PUT = re.compile(r'^pocket:put\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*(.+)$')
R_POCKET_MOVE = re.compile(r'^pocket:move\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)$')

# ---- Extended Edition: fun. / text. / edm. / oma. / aud. / fun.ag / deb. ----
R_FUN_PRINT = re.compile(
    r'^fun\.text/print\((.+?)\)\s*(?:/\(\s*([\d,\.]+)\s*[x\u00d7]\s*([\d,\.]+)\s*\))?$')
R_TEXT_STYLE = re.compile(r'^text\.set-style/\((.+)\)$')
R_TEXT_RAINBOW = re.compile(r'^text\.rainbowpalette/animation$')
R_TEXT_ANIM = re.compile(r'^text\.animation/set animation$')

R_EDM_EDIT = re.compile(r'^edm\.pro\.:edit(?:\s+(.+))?$')
R_EDM_TRANS = re.compile(r'^edm\.:next/transition\s+(.+)$')
R_EDM_STOP = re.compile(r'^edm\.:next/stop\s+(.+)$')
R_EDM_RESUME = re.compile(r'^edm\.:next/resume\s+(.+)$')

R_APP_START = re.compile(r'^oma\.?:?appbuild/start\.?(?:\s+(.+))?$')
R_APP_NEXT = re.compile(r'^oma\.?:?//app/next(?:\s+(.+))?$')
R_APP_STOP = re.compile(r'^oma\.?:?//app/stop(?:\s+(.+))?$')
R_APP_PAUSE = re.compile(r'^oma\.?:?//app/pause(?:\s+(.+))?$')
R_APP_CACHE = re.compile(r'^oma\.?:?//app/cache(?:\s+(.+))?$')
R_WEB_BUILD = re.compile(r'^oma\.?:?(?://)?(?:webbuild|web)/start\s+(.+)$')
R_WEB_AUTH = re.compile(r'^oma\.?:?//web/auth\s+(.+)$')
R_WEB_NEXT = re.compile(r'^oma\.?:?//web/next(?:\s+(.+))?$')
R_WEB_END = re.compile(r'^oma\.?:?//web/end(?:\s+(.+))?$')
R_WEB_FORMAT = re.compile(r'^oma\.?:?//web/format;start(?:\s+(.+))?$')
R_WEB_DESIGNFMT = re.compile(r'^oma\.?:?//web/design-format;start(?:\s+(.+))?$')
R_WEB_FINDSKILL = re.compile(r'^oma\.?:?//web/design;\s*find-design-skill\s+(.+)$')
R_NET_START = re.compile(r'^oma\.?:?//internet;search/start\s+(.+)$')
R_NET_END = re.compile(r'^oma\.?:?//internet;search/end(?:\s+(.+))?$')
R_AUD_COMBINE = re.compile(r'^oma\.?:?//music;audio//\(combine-audio-file\)\s+(.+)$')
R_AUD_ENHANCE = re.compile(r'^oma\.?:?//music;audio//\(enhance\)\s+(.+)$')
R_AUD_LOOP = re.compile(r'^oma\.?:?//music;audio/\(set-loop\)\s+(.+)$')
R_AUD_REVERB = re.compile(
    r'^oma\.?:?//music;audio//\(reverb\)\s+(.+?)\s*=\s*SET\s+INTENSITY\s+(\d+)$', re.I)

R_AG_CREATE = re.compile(r'^(?:fun\.ag|deb)\.?:?//agent\s+([A-Za-z_]\w*)$')
R_AG_START = re.compile(r'^(?:fun\.ag|deb)\.?:?//agent;start\s+([A-Za-z_]\w*)$')
R_AG_FLOW = re.compile(r'^(?:fun\.ag|deb)\.?:?//agentflow\.\s*([A-Za-z_]\w*)$')
R_AG_POWER = re.compile(r'^(?:fun\.ag|deb)\.?:?//agentflow;\s*x\s*=\s*(.+)$')
R_AG_SPEED = re.compile(r'^(?:fun\.ag|deb)\.?:?//agentflow;\s*work-speed-x\s*=\s*(.+)$')

R_AUD_TONE = re.compile(r'^aud\.:tone\s+([A-Za-z_]\w*)/\(([^;)]+);\s*([\d.]+)\)$')
R_AUD_PLAY = re.compile(r'^aud\.:play\s+([A-Za-z_]\w*)(?:\s+"([^"]+)")?$')

BLOCK_KINDS = ('model', 'def', 'repeat', 'store_start', 'ag_flow')


def classify(s):
    for pat, kind in ((R_END, 'end'), (R_ELSE, 'else'), (R_IF, 'if'),
                      (R_REPEAT, 'repeat'), (R_MODEL, 'model'), (R_DEF, 'def'),
                      (R_STORE_START, 'store_start'), (R_STORE_MODE, 'store_mode'),
                      (R_STATUS, 'status'), (R_RUN, 'run'),
                      (R_TR, 'dist'), (R_POCKET_ADD, 'pocket_add'),
                      (R_POCKET_PUT, 'pocket_put'), (R_POCKET_MOVE, 'pocket_move'),
                      (R_FUN_PRINT, 'fun_print'), (R_TEXT_STYLE, 'text_style'),
                      (R_TEXT_RAINBOW, 'text_rainbow'), (R_TEXT_ANIM, 'text_anim'),
                      (R_EDM_EDIT, 'edm_edit'), (R_EDM_TRANS, 'edm_transition'),
                      (R_EDM_STOP, 'edm_stop'), (R_EDM_RESUME, 'edm_resume'),
                      (R_APP_START, 'oma_app_start'), (R_APP_NEXT, 'oma_app_next'),
                      (R_APP_STOP, 'oma_app_stop'), (R_APP_PAUSE, 'oma_app_pause'),
                      (R_APP_CACHE, 'oma_app_cache'),
                      (R_WEB_BUILD, 'oma_web_build'), (R_WEB_AUTH, 'oma_web_auth'),
                      (R_WEB_NEXT, 'oma_web_next'), (R_WEB_END, 'oma_web_end'),
                      (R_WEB_FORMAT, 'oma_web_format'),
                      (R_WEB_DESIGNFMT, 'oma_web_designfmt'),
                      (R_WEB_FINDSKILL, 'oma_web_findskill'),
                      (R_NET_START, 'oma_net_start'), (R_NET_END, 'oma_net_end'),
                      (R_AUD_COMBINE, 'aud_combine'), (R_AUD_ENHANCE, 'aud_enhance'),
                      (R_AUD_LOOP, 'aud_loop'), (R_AUD_REVERB, 'aud_reverb'),
                      (R_AG_CREATE, 'ag_create'), (R_AG_START, 'ag_start'),
                      (R_AG_FLOW, 'ag_flow'), (R_AG_POWER, 'ag_power'),
                      (R_AG_SPEED, 'ag_speed'),
                      (R_AUD_TONE, 'aud_tone'), (R_AUD_PLAY, 'aud_play'),
                      (R_LET, 'let'), (R_OUT, 'out'), (R_GIVE, 'give')):
        m = pat.match(s)
        if m:
            return {'kind': kind, 'm': m, 'text': s}
    return {'kind': 'expr', 'm': None, 'text': s}


def _strip_comment(s):
    in_str = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            in_str = not in_str
        elif ch == '#' and not in_str and (i == 0 or s[i - 1] in ' \t'):
            return s[:i].rstrip()
        i += 1
    return s


def parse_program(text):
    lines = []
    for n, raw in enumerate(text.splitlines(), 1):
        s = _strip_comment(raw.strip())
        if not s or s.startswith('#'):
            continue
        lines.append((n, s))
    stmts, i = parse_seq(lines, 0, None)
    if i != len(lines):
        raise OmarisError("line %d: unexpected %r" % (lines[i][0], lines[i][1]))
    return stmts


def parse_seq(lines, i, stoppers):
    out = []
    while i < len(lines):
        n, s = lines[i]
        c = classify(s)
        if stoppers and c['kind'] in stoppers:
            return out, i
        i += 1
        if c['kind'] == 'if':
            body, i = parse_seq(lines, i, ('else', 'end'))
            els = []
            if i < len(lines) and classify(lines[i][1])['kind'] == 'else':
                i += 1
                els, i = parse_seq(lines, i, ('end',))
            _require_end(lines, i)
            i += 1
            out.append(('if', c, body, els))
        elif c['kind'] in BLOCK_KINDS:
            body, i = parse_seq(lines, i, ('end',))
            _require_end(lines, i)
            i += 1
            out.append((c['kind'], c, body))
        else:
            out.append((c['kind'], c))
    if stoppers:
        raise OmarisError("unterminated block: missing 'end' (opened near line %d)"
                          % (lines[-1][0] if lines else 0))
    return out, i


def _require_end(lines, i):
    if i >= len(lines) or classify(lines[i][1])['kind'] != 'end':
        ln = lines[i][0] if i < len(lines) else (lines[-1][0] if lines else 0)
        raise OmarisError("line %d: missing 'end' for block" % ln)


# ---------------------------------------------------------------------------
# Runtime values and environments
# ---------------------------------------------------------------------------

class Env:
    """Variable environment. Blocks share the enclosing scope; only function
    calls create a fresh scope that falls back to global variables."""

    def __init__(self, rt, vars=None, parent=True):
        self.rt = rt
        self.vars = vars if vars is not None else {}
        self.use_globals = parent

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.use_globals and name in self.rt.globals:
            return self.rt.globals[name]
        if name in BUILTINS and not callable(BUILTINS[name]):
            return BUILTINS[name]
        raise OmarisError("unknown variable %r" % name)

    def set(self, name, value):
        self.vars[name] = value


class Fun:
    def __init__(self, name, params, body):
        self.name, self.params, self.body = name, params, body


# ---------------------------------------------------------------------------
# Builtins
# ---------------------------------------------------------------------------

def _cumsum(xs):
    out, s = [], 0
    for v in xs:
        s = s + v
        out.append(s)
    return out


def _diff(xs):
    if len(xs) < 2:
        return []
    return [xs[k + 1] - xs[k] for k in range(len(xs) - 1)]


def _fmt(v):
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


BUILTINS = {
    'sum': lambda *a: sum(a[0]) if len(a) == 1 and isinstance(a[0], list) else sum(a),
    'mean': lambda xs: sum(xs) / len(xs),
    'min': lambda xs: min(xs), 'max': lambda xs: max(xs),
    'abs': abs, 'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'floor': math.floor, 'ceil': math.ceil, 'round': round,
    'len': len, 'range': lambda *a: list(range(*a)),
    'str': _fmt, 'num': lambda v: float(v) if '.' in str(v) else int(v),
    'upper': lambda s: s.upper(), 'lower': lambda s: s.lower(),
    'cumsum': _cumsum, 'diff': _diff, 'diff2': lambda xs: _diff(_diff(xs)),
    'true': True, 'false': False,
}


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Interp:
    def __init__(self):
        self.globals = {}
        self.functions = {}
        self.models = {}
        self.stores = {}
        self.store_mode = None       # 'open' (local device) / 'closed' (host)
        self._item_seq = {}
        # --- Extended Edition state ---
        self.builds = {}             # app / web build operations
        self.audio = {}              # named audio buffers
        self.agents = {}             # fun.ag agents
        self.disabled = set()        # functions stopped by edm.:next/stop
        self.sandboxes = set()       # builds turned into sandboxes (edm.pro.:edit)
        self.page = []               # fun.text layout log: (x, y, text)
        self.text_style = ''
        self.rainbow = False
        self.animate = False
        self._anim = 0
        self._active_agent = None

    # -- expression evaluation ------------------------------------------------

    def eval(self, node, env):
        k = node[0]
        if k == 'num':
            return node[1]
        if k == 'str':
            return node[1]
        if k == 'var':
            return env.get(node[1])
        if k == 'list':
            return [self.eval(x, env) for x in node[1]]
        if k == 'dict':
            return {key: self.eval(v, env) for key, v in node[1]}
        if k == 'idx':
            return self.eval(node[1], env)[int(self.eval(node[2], env))]
        if k == 'bin':
            return self._bin(node[1], self.eval(node[2], env), self.eval(node[3], env))
        if k == 'and':
            return bool(self.eval(node[1], env)) and bool(self.eval(node[2], env))
        if k == 'or':
            return bool(self.eval(node[1], env)) or bool(self.eval(node[2], env))
        if k == 'not':
            return not self.eval(node[1], env)
        if k == 'neg':
            return -self.eval(node[1], env)
        if k == 'uop':
            return self._uop(node[1], self.eval(node[2], env))
        if k == 'call':
            return self._call(node[1], [self.eval(a, env) for a in node[2]], env)
        raise OmarisError("internal: unknown AST node %r" % (k,))

    @staticmethod
    def _bin(op, a, b):
        if op == '+':
            if isinstance(a, str) or isinstance(b, str):
                if isinstance(a, str) and isinstance(b, str):
                    return a + b
                raise OmarisError("'+' between text and number: wrap the number in str(...)")
            return a + b
        if op == '-':
            return a - b
        if op == '*':
            if isinstance(a, str) and isinstance(b, (int, float)):
                return a * int(b)
            return a * b
        if op == '/':
            if b == 0:
                raise OmarisError("division by zero")
            return a / b
        if op == '%':
            return a % b
        if op == '**':
            return a ** b
        if op == '==':
            return a == b
        if op == '!=':
            return a != b
        if op == '<':
            return a < b
        if op == '<=':
            return a <= b
        if op == '>':
            return a > b
        if op == '>=':
            return a >= b
        raise OmarisError("internal: unknown operator %r" % op)

    def _uop(self, symbol, v):
        if not isinstance(v, list):
            raise OmarisError("calculus operator %s needs a list (e.g. a series of values); got %s"
                              % (symbol, _fmt(v)))
        if symbol == '\u222b':          # ∫  cumulative total
            return _cumsum(v)
        if symbol == '\u222b\u222b':    # ∫∫ cumulative of cumulative
            return _cumsum(_cumsum(v))
        if symbol == '\u222b\u222b\u222b':
            return _cumsum(_cumsum(_cumsum(v)))
        if symbol == '\u1e8f':          # ẏ first difference (rate of change)
            return _diff(v)
        if symbol == '\u00ff':          # ÿ second difference (acceleration)
            return _diff(_diff(v))
        raise OmarisError("internal: unknown calculus operator %r" % symbol)

    def _call(self, name, args, env):
        fn = self.functions.get(name)
        if fn is not None and name in self.disabled:
            raise OmarisError("code is not live: %r was stopped with edm.:next/stop "
                              "(resume it with edm.:next/resume)" % name)
        if fn is None and '.' in name:
            mname, fname = name.split('.', 1)
            model = self.models.get(mname)
            if model and fname in model['functions']:
                fn = model['functions'][fname]
        if fn is not None:
            if len(args) != len(fn.params):
                raise OmarisError("%s expects %d argument(s), got %d"
                                  % (name, len(fn.params), len(args)))
            call_env = Env(self, dict(zip(fn.params, args)))
            try:
                self.exec_stmts(fn.body, call_env)
            except ReturnSignal as r:
                return r.value
            return None
        if name in BUILTINS:
            return BUILTINS[name](*args)
        raise OmarisError("unknown function %r" % name)

    # -- statement execution --------------------------------------------------

    def exec_stmts(self, stmts, env):
        for st in stmts:
            self.exec_stmt(st, env)

    def exec_stmt(self, st, env):
        kind, c = st[0], st[1]

        if kind == 'let':
            env.set(c['m'].group(1), self.eval(ExprParser(c['m'].group(2)).parse(), env))
        elif kind == 'out':
            print(_fmt(self.eval(ExprParser(c['m'].group(1)).parse(), env)))
        elif kind == 'give':
            expr = c['m'].group(1)
            raise ReturnSignal(self.eval(ExprParser(expr).parse(), env) if expr else None)
        elif kind == 'expr':
            self.eval(ExprParser(c['text']).parse(), env)
        elif kind == 'if':
            if self.eval(ExprParser(c['m'].group(1)).parse(), env):
                self.exec_stmts(st[2], env)
            else:
                self.exec_stmts(st[3], env)
        elif kind == 'repeat':
            seq = self.eval(ExprParser(c['m'].group(1)).parse(), env)
            var = c['m'].group(2)
            items = range(1, int(seq) + 1) if isinstance(seq, (int, float)) else seq
            for item in items:
                if var:
                    env.set(var, item)
                self.exec_stmts(st[2], env)
        elif kind == 'model':
            self._model(st, env)
        elif kind == 'def':
            parts = [p.strip() for p in c['m'].group(1).split(';')]
            name = parts[0]
            params = [p for p in parts[1:] if p]
            fn = Fun(name, params, st[2])
            if getattr(self, '_current_model', None):
                self.models[self._current_model]['functions'][name] = fn
            else:
                self.functions[name] = fn
        elif kind == 'store_mode':
            self.store_mode = c['m'].group(1).lower()
        elif kind == 'store_start':
            self._store_start(st, c, env)
        elif kind == 'status':
            self._status(c['m'].group(1))
        elif kind == 'run':
            self._store_run(c['m'].group(1), env)
        elif kind == 'dist':
            self._dist(c, env)
        elif kind == 'pocket_add':
            self._store(c['m'].group(1))['pockets'].setdefault(c['m'].group(2), {})
        elif kind == 'pocket_put':
            store = self._store(c['m'].group(1))
            self._pocket(store, c['m'].group(2))[c['m'].group(3)] = \
                self.eval(ExprParser(c['m'].group(4)).parse(), env)
            self._reset_weight(store, "pocket:put %s into %s"
                               % (c['m'].group(3), c['m'].group(2)))
        elif kind == 'pocket_move':
            store = self._store(c['m'].group(1))
            src, dst, key = c['m'].group(2), c['m'].group(3), c['m'].group(4)
            self._pocket(store, dst)[key] = self._pocket(store, src).pop(key)
            self._reset_weight(store, "pocket:move %s: %s -> %s" % (key, src, dst))

        # ---------------- Extended Edition handlers ----------------
        elif kind == 'fun_print':
            self._fun_print(c['m'], env)
        elif kind == 'text_style':
            self.text_style = c['m'].group(1).strip()
            print("[text style set: %s]" % self.text_style)
        elif kind == 'text_rainbow':
            self.rainbow = True
            print("[rainbow palette armed — combine with text.animation/set animation]")
        elif kind == 'text_anim':
            self.animate = True
            print("[text animation enabled]")
        elif kind == 'edm_edit':
            ref = (c['m'].group(1) or '').strip()
            self.sandboxes.add(ref)
            print("edm.pro.:edit — %r is now a functional sandbox" % (ref or 'current build'))
        elif kind == 'edm_transition':
            self._edm_transition(c['m'].group(1))
        elif kind == 'edm_stop':
            target = c['m'].group(1).strip()
            if target not in self.functions:
                raise OmarisError("edm.:next/stop: %r is not a function" % target)
            self.disabled.add(target)
            print("edm.:next/stop — %r disabled (no longer live)" % target)
        elif kind == 'edm_resume':
            target = c['m'].group(1).strip()
            self.disabled.discard(target)
            print("edm.:next/resume — %r is live again" % target)
        elif kind == 'oma_app_start':
            names = (c['m'].group(1) or 'app').split()
            for nm in names:
                self._start_build(nm.strip(), 'app',
                                  ['scaffold', 'interface', 'logic', 'assets', 'release'])
        elif kind in ('oma_app_next', 'oma_app_stop', 'oma_app_pause', 'oma_app_cache'):
            self._app_op(kind, (c['m'].group(1) or '').strip())
        elif kind == 'oma_web_build':
            self._start_build(c['m'].group(1).strip(), 'web',
                              ['wireframe', 'database', 'frontend', 'backend', 'launch'])
        elif kind == 'oma_web_auth':
            self._web_auth(c['m'].group(1).strip())
        elif kind == 'oma_web_next':
            self._web_op((c['m'].group(1) or '').strip(), 'next')
        elif kind == 'oma_web_end':
            self._web_op((c['m'].group(1) or '').strip(), 'end')
        elif kind == 'oma_web_format':
            self._web_op((c['m'].group(1) or '').strip(), 'format-wireframe-database')
        elif kind == 'oma_web_designfmt':
            self._web_op((c['m'].group(1) or '').strip(), 'design-format')
        elif kind == 'oma_web_findskill':
            self._find_design_skill(c['m'].group(1).strip())
        elif kind == 'oma_net_start':
            self._internet_search(c['m'].group(1).strip(), True)
        elif kind == 'oma_net_end':
            self._internet_search((c['m'].group(1) or '').strip(), False)
        elif kind == 'aud_combine':
            self._audio_combine(c['m'].group(1))
        elif kind == 'aud_enhance':
            self._audio_enhance(c['m'].group(1).strip())
        elif kind == 'aud_loop':
            self._audio_loop(c['m'].group(1).strip())
        elif kind == 'aud_reverb':
            self._audio_reverb(c['m'].group(1).strip(), int(c['m'].group(2)))
        elif kind == 'ag_create':
            self._agent_create(c['m'].group(1))
        elif kind == 'ag_start':
            self._agent_start(c['m'].group(1), env)
        elif kind == 'ag_flow':
            self._agent_flow(c['m'].group(1), st[2])
        elif kind == 'ag_power':
            self._agent_tune('power', self.eval(ExprParser(c['m'].group(1)).parse(), env))
        elif kind == 'ag_speed':
            self._agent_tune('speed', self.eval(ExprParser(c['m'].group(1)).parse(), env))
        elif kind == 'aud_tone':
            self._tone(c['m'].group(1), c['m'].group(2).strip(), float(c['m'].group(3)))
        elif kind == 'aud_play':
            self._play(c['m'].group(1), c['m'].group(2))
        else:
            raise OmarisError("internal: unknown statement kind %r" % kind)

    def _model(self, st, env):
        name = st[1]['m'].group(1)
        if name in self.models:
            raise OmarisError("model %r already exists" % name)
        self.models[name] = {'name': name, 'functions': {}}
        self._current_model = name
        try:
            self.exec_stmts(st[2], env)
        finally:
            self._current_model = None

    # -- storage containers ---------------------------------------------------

    def _store(self, name):
        if name not in self.stores:
            raise OmarisError("storage container %r does not exist "
                              "(declare it with store.fun.;start)" % name)
        return self.stores[name]

    @staticmethod
    def _pocket(store, name):
        if name not in store['pockets']:
            raise OmarisError("pocket %r does not exist in %r (add it with pocket:add)"
                              % (name, store['name']))
        return store['pockets'][name]

    def _reset_weight(self, store, why):
        """The heart of store.fun.;start: whenever code or data enters the
        container (or shifts between hidden pockets), the tracked value —
        line count + weight — resets to 0 while the code itself remains."""
        store['events'].append("%s: %d lines / %d bytes -> tracked reset to 0"
                               % (why, store['actual_lines'], store['actual_bytes']))
        store['tracked_weight'] = 0

    def _store_start(self, st, c, env):
        m = c['m']
        name = m.group(5) or m.group(1) or ('store_%d' % (len(self.stores) + 1))
        if name in self.stores:
            raise OmarisError("storage container %r already exists" % name)
        if m.group(2):
            cap = (m.group(2), int(m.group(3)))
            limit = cap[1] * (1024 ** 3 if cap[0] == 'GB' else 1024 ** 4)
        elif m.group(4):
            cap, limit = 'Infinite', None
        else:
            cap, limit = 'Infinite', None
        src_lines = [body[1]['text'] for body in st[2]]
        src_lines.insert(0, c['text'])
        bytes_ = sum(len(l.encode('utf-8')) + 1 for l in src_lines)
        if limit is not None and bytes_ > limit:
            raise OmarisError("capacity exceeded: %s needs %d bytes but %s(%d) allows %d"
                              % (name, bytes_, cap[0], cap[1], limit))
        store = {
            'name': name, 'capacity': cap, 'mode': self.store_mode,
            'tracked_weight': 0, 'actual_lines': len(src_lines),
            'actual_bytes': bytes_, 'pockets': {'root': {}},
            'code': st[2], 'events': [],
        }
        self.stores[name] = store
        self._reset_weight(store, "store.fun.;start %s absorbed %d line(s) of code"
                           % (name, len(st[2])))

    def _status(self, name):
        s = self._store(name)
        cap = ('capacity: %s(%d)' % (s['capacity'][0], s['capacity'][1])
               if isinstance(s['capacity'], tuple) else 'capacity: Infinite')
        print("store %r  [%s]  %s" % (name, s['mode'] or 'unspecified', cap))
        print("  tracked weight : %d   (true size: %d lines / %d bytes, retained)"
              % (s['tracked_weight'], s['actual_lines'], s['actual_bytes']))
        print("  pockets        : %s" % ', '.join(s['pockets']))
        print("  events         : %d logged" % len(s['events']))
        if s['events']:
            print("  last event     : %s" % s['events'][-1])

    def _store_run(self, name, env):
        s = self._store(name)
        self.exec_stmts(s['code'], Env(self))
        self._reset_weight(s, "store.fun.;run %s (code re-entered runtime)" % name)

    def _dist(self, c, env):
        mode, rest = c['m'].group(1), c['m'].group(2)
        parts = rest.split('->')
        if len(parts) != 2:
            raise OmarisError("distribution needs the form '<source> -> <destination>'")
        src, dst = parts[0].strip(), parts[1].strip()

        def resolve(ref):
            if '.' in ref:
                cname, pname = ref.split('.', 1)
                return self._store(cname), pname
            return self._store(ref), None

        dst_store, dst_pocket = resolve(dst)
        if mode.startswith('dist.;'):
            value = self.eval(ExprParser(src).parse(), env)
            self._item_seq[dst_store['name']] = self._item_seq.get(dst_store['name'], 0) + 1
            pocket = dst_store['pockets'].setdefault(dst_pocket or 'root', {})
            pocket['item_%d' % self._item_seq[dst_store['name']]] = value
            self._reset_weight(dst_store, "tr.fun.:(dist.; value) delivered an item")
            return
        src_store, src_pocket = resolve(src)
        moved = 0
        if src_pocket:
            dst_store['pockets'].setdefault(dst_pocket or src_pocket, {}).update(
                self._pocket(src_store, src_pocket))
            self._pocket(src_store, src_pocket).clear()
            moved = 1
        else:
            for pname, entries in src_store['pockets'].items():
                dst_store['pockets'].setdefault(dst_pocket or pname, {}).update(entries)
                entries.clear()
                moved += 1
        self._reset_weight(src_store, "tr.fun.:(dist.) shipped %d pocket(s) out" % moved)
        self._reset_weight(dst_store, "tr.fun.:(dist.) received %d pocket(s)" % moved)

    # ---------------- Extended Edition runtime ----------------

    _ANSI = {'bold': '1', 'italic': '3', 'underline': '4',
             'red': '31', 'green': '32', 'yellow': '33', 'blue': '34',
             'magenta': '35', 'cyan': '36', 'white': '37'}

    def _ansi(self, text):
        codes = []
        for part in self.text_style.split('+'):
            code = self._ANSI.get(part.strip().lower())
            if code:
                codes.append(code)
        pre = ''.join('\x1b[%sm' % c for c in codes)
        return pre + text + ('\x1b[0m' if pre else '')

    @staticmethod
    def _rainbow(text, offset=0):
        out = []
        for i, ch in enumerate(text):
            h = (i * 24 + offset * 40) % 360 / 60.0
            zone, frac = int(h) % 6, h - int(h)
            table = [(255, int(255 * frac), 0), (int(255 * (1 - frac)), 255, 0),
                     (0, 255, int(255 * frac)), (0, int(255 * (1 - frac)), 255),
                     (int(255 * frac), 0, 255), (255, 0, int(255 * (1 - frac)))]
            r, g, b = table[zone]
            out.append('\x1b[38;2;%d;%d;%dm%s' % (r, g, b, ch))
        return ''.join(out) + '\x1b[0m'

    def _fun_print(self, m, env):
        value = _fmt(self.eval(ExprParser(m.group(1)).parse(), env))
        x, y = m.group(2), m.group(3)
        if x and y:
            x, y = float(x.replace(',', '')), float(y.replace(',', ''))
            self.page.append((x, y, value))
            indent = ' ' * max(0, min(78, int(x / 64)))
            body = self._rainbow(value) if getattr(self, 'animate', False) and self.rainbow \
                else self._ansi(value)
            print(indent + body + '   [placed at %d x %d]' % (x, y))
            self._anim = getattr(self, '_anim', 0) + 1
        else:
            print(self._ansi(value))

    def _start_build(self, name, bkind, phases):
        if name in self.builds and self.builds[name]['state'] == 'running':
            raise OmarisError("build %r is already running" % name)
        self.builds[name] = {'kind': bkind, 'phase': 0, 'phases': phases,
                             'state': 'running', 'auth': set(), 'log': []}
        verb = 'application' if bkind == 'app' else 'website'
        print("oma.: %s build started: %r  (phase 1/%d: %s)"
              % (verb, name, len(phases), phases[0]))

    def _one_build(self, name, what):
        if name:
            if name not in self.builds:
                raise OmarisError("no build named %r (start one with oma.:appbuild/start "
                                  "or oma.://webbuild/start)" % name)
            return self.builds[name], name
        running = [(n, b) for n, b in self.builds.items() if b['state'] == 'running']
        if not running:
            raise OmarisError("no running %s build" % what)
        return running[0][1], running[0][0]

    def _app_op(self, kind, name):
        build, name = self._one_build(name, 'app')
        if kind == 'oma_app_next':
            if build['state'] == 'paused':
                build['state'] = 'running'
                print("oma.://app — build %r resumed" % name)
            if build['state'] != 'running':
                raise OmarisError("build %r is %s; it cannot advance" % (name, build['state']))
            build['phase'] += 1
            if build['phase'] >= len(build['phases']):
                build['state'] = 'done'
                print("oma.://app — build %r complete: app is live" % name)
            else:
                print("oma.://app/next — %r advanced to phase %d/%d: %s"
                      % (name, build['phase'] + 1, len(build['phases']),
                         build['phases'][build['phase']]))
        elif kind == 'oma_app_stop':
            build['state'] = 'stopped'
            print("oma.://app/stop — build %r stopped" % name)
        elif kind == 'oma_app_pause':
            build['state'] = 'paused'
            print("oma.://app/pause — build %r paused" % name)
        elif kind == 'oma_app_cache':
            cname = 'app_cache_%s' % name
            if cname not in self.stores:
                self.stores[cname] = {'name': cname, 'capacity': 'Infinite',
                                      'mode': self.store_mode, 'tracked_weight': 0,
                                      'actual_lines': 1, 'actual_bytes': 0,
                                      'pockets': {'root': {}}, 'code': [], 'events': []}
            snapshot = {'app': name, 'phase': build['phase'],
                        'state': build['state'], 'log': list(build['log'])}
            self.stores[cname]['pockets']['root']['snapshot'] = snapshot = dict(snapshot)
            self._reset_weight(self.stores[cname], "oma.://app/cache cached build %r" % name)
            print("oma.://app/cache — entire app %r cached into storage container %r" % (name, cname))

    def _web_auth(self, source):
        webs = [n for n, b in self.builds.items() if b['kind'] == 'web']
        if not webs:
            raise OmarisError("oma.://web/auth: no website build (start one with "
                              "oma.://webbuild/start)")
        target = webs[-1]
        self.builds[target]['auth'].add(source)
        print("oma.://web/auth — authentication request sent for source %r (site %r)"
              % (source, target))

    def _web_op(self, name, what):
        build, name = self._one_build(name, 'web')
        build['log'].append(what)
        labels = {'auth-source': 'authentication request sent for source',
                  'next': 'advanced to next coding operation',
                  'end': 'website coding operation ended',
                  'format-wireframe-database': 'formatting wireframe and database',
                  'design-format': 'design operation started'}
        if what == 'end':
            build['state'] = 'done'
        print("oma.://web — %r: %s %s" % (name, labels.get(what, what),
                                          '' if what != 'auth-source' else '(authenticated)'))

    DESIGN_SKILLS = ['minimal-grid', 'glassmorphism', 'swiss-typography',
                     'brutalist', 'material-motion', 'neo-skeuomorph']

    def _find_design_skill(self, query):
        q = query.lower()
        hits = [s for s in self.DESIGN_SKILLS
                if any(tok in s for tok in q.replace('-', ' ').split())] or self.DESIGN_SKILLS
        print("oma.://web/design — found %d design skill(s) for %r: %s"
              % (len(hits), query, ', '.join(hits)))

    def _internet_search(self, query, start):
        if start:
            opened = False
            if query:
                try:
                    webbrowser.open('https://duckduckgo.com/?q=' + urllib.parse.quote(query))
                    opened = True
                except Exception:
                    opened = False
            print("oma.://internet;search — started search on the user's browser: %r%s"
                  % (query, '' if opened else ' (browser unavailable; query logged)'))
        else:
            print("oma.://internet;search — search ended")

    def _audio(self, name):
        if name not in self.audio:
            raise OmarisError("audio buffer %r does not exist (create one with aud.:tone)" % name)
        return self.audio[name]

    def _audio_combine(self, rest):
        parts = [p.strip() for p in rest.split('->')]
        names = parts[0].split()
        out_name = parts[1].strip() if len(parts) > 1 else 'combined'
        data = []
        for nm in names:
            data.extend(self._audio(nm)['samples'])
        self.audio[out_name] = {'rate': 44100, 'samples': data}
        print("oma.://music;audio — combined %d file(s) -> %r (%d samples)"
              % (len(names), out_name, len(data)))

    def _audio_enhance(self, name):
        buf = self._audio(name)
        peak = max((abs(s) for s in buf['samples']), default=0) or 1.0
        buf['samples'] = [s * 0.9 / peak for s in buf['samples']]
        print("oma.://music;audio — enhanced %r (normalized to 0.9 peak)" % name)

    def _audio_loop(self, rest):
        parts = rest.split()
        name, n = parts[0], int(parts[1]) if len(parts) > 1 else 2
        buf = self._audio(name)
        buf['samples'] = buf['samples'] * n
        print("oma.://music;audio — loop set: %r now repeats %dx" % (name, n))

    def _audio_reverb(self, name, intensity):
        buf = self._audio(name)
        xs, delay = buf['samples'], int(44100 * 0.15)
        out = list(xs)
        for i in range(delay, len(xs)):
            out[i] += (intensity / 100.0) * xs[i - delay]
        buf['samples'] = out
        print("oma.://music;audio — reverb on %r at intensity %d" % (name, intensity))

    def _tone(self, name, freq, ms):
        rate = 44100
        n = int(rate * ms / 1000.0)
        f = float(freq)
        self.audio[name] = {'rate': rate,
                            'samples': [0.5 * math.sin(2 * math.pi * f * k / rate)
                                        for k in range(n)]}
        print("aud.:tone — buffer %r: %s Hz for %g ms (%d samples)" % (name, freq, ms, n))

    def _play(self, name, path):
        buf = self._audio(name)
        path = os.path.abspath(path if path else (name + '.wav'))
        with wave.open(path, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(buf['rate'])
            frames = b''.join(
                struct.pack('<h', max(-32768, min(32767, int(s * 32767))))
                for s in buf['samples'])
            w.writeframes(frames)
        print("aud.:play — rendered %r to %s (%d samples, %.2fs)"
              % (name, path, len(buf['samples']), len(buf['samples']) / buf['rate']))

    def _agent_create(self, name):
        self.agents[name] = {'power': 1, 'speed': 0, 'state': 'idle',
                             'workflow': [], 'ticks': 0}
        self._active_agent = name
        print("fun.ag.://agent — agent %r created (power 1, work-speed 0)" % name)

    def _agent(self, name):
        if name not in self.agents:
            raise OmarisError("agent %r does not exist (create one with fun.ag.://agent)" % name)
        return self.agents[name]

    def _agent_flow(self, name, body):
        agent = self._agent(name)
        agent['workflow'] = body
        self._active_agent = name
        print("fun.ag.://agentflow. — workflow with %d step(s) assigned to agent %r"
              % (len(body), name))
        if agent['state'] == 'working':
            self._run_workflow(name, agent)

    def _agent_start(self, name, env):
        agent = self._agent(name)
        agent['state'] = 'working'
        self._active_agent = name
        print("fun.ag.://agent;start — agent %r started (power %s, work-speed %s)"
              % (name, agent['power'], agent['speed']))
        if agent['workflow']:
            self._run_workflow(name, agent, env)
        else:
            print("  (no workflow assigned yet — use fun.ag.://agentflow. %s ... end)" % name)

    def _run_workflow(self, name, agent, env=None):
        ticks = 0
        run_env = env if env is not None else Env(self)
        for stt in agent['workflow']:
            self.exec_stmt(stt, run_env)
            ticks += 1
        agent['ticks'] += ticks
        agent['state'] = 'idle'
        print("  agent %r completed %d step(s) — power %s, work-speed %s"
              % (name, ticks, agent['power'], agent['speed']))

    def _agent_tune(self, what, value):
        if not self._active_agent:
            raise OmarisError("no active agent (create one with fun.ag.://agent)")
        agent = self._agent(self._active_agent)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise OmarisError("agent %s value must be a number" % what)
        agent[what] = value
        if what == 'speed' and agent['speed'] >= agent['power']:
            raise OmarisError(
                "NOTICE: work speed value must be less than power value "
                "(agent %r: work-speed %s >= power %s)" % (self._active_agent, agent['speed'], agent['power']))
        label = 'power (x)' if what == 'power' else 'work-speed (work-speed-x)'
        print("fun.ag.://agentflow — %s of agent %r increased to %s"
              % (label, self._active_agent, value))

    def _edm_transition(self, rest):
        parts = rest.split('->')
        if len(parts) != 2:
            raise OmarisError("edm.:next/transition needs '<source> -> <destination>'")
        src, dst = parts[0].strip(), parts[1].strip()
        if src in self.functions:
            store = self._store(dst)
            self._pocket(store, 'root')['fn_' + src] = self.functions.pop(src)
            self._reset_weight(store, "edm.:next/transition moved function %r into storage" % src)
            print("edm.:next/transition — function %r moved into container %r" % (src, dst))
            return
        sstore, spocket = ((src.split('.') + [None])[:2] if '.' in src else (src, None))
        dstore, dpocket = ((dst.split('.') + [None])[:2] if '.' in dst else (dst, None))
        s = self._store(sstore)
        d = self._store(dstore)
        moved = 0
        if spocket:
            d['pockets'].setdefault(dpocket or spocket, {}).update(self._pocket(s, spocket))
            self._pocket(s, spocket).clear()
            moved = 1
        else:
            for pname, entries in list(s['pockets'].items()):
                d['pockets'].setdefault(dpocket or pname, {}).update(entries)
                entries.clear()
                moved += 1
        self._reset_weight(s, "edm.:next/transition shipped code out")
        self._reset_weight(d, "edm.:next/transition received code")
        print("edm.:next/transition — code moved %s -> %s (%d pocket(s))" % (src, dst, moved))

    # -- top level ------------------------------------------------------------


    def run(self, text):
        self.exec_stmts(parse_program(text), Env(self))


# ---------------------------------------------------------------------------
# REPL + CLI
# ---------------------------------------------------------------------------

_OPENERS = (R_IF, R_REPEAT, R_MODEL, R_DEF, R_STORE_START)


def _depth_delta(line):
    c = classify(line.strip())
    d = 0
    if c['kind'] in ('if', 'repeat', 'model', 'def', 'store_start') or c['kind'] in BLOCK_KINDS:
        d += 1
    if c['kind'] in ('end', 'else'):
        d -= 1
    return d


def repl():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print("Omaris %s — interactive REPL  (ctrl+z then enter, or ctrl+c to exit)" % VERSION)
    rt = Interp()
    buf, depth = [], 0
    while True:
        prompt = 'omaris> ' if depth == 0 else '......> '
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line.strip() and depth == 0:
            continue
        buf.append(line)
        depth += _depth_delta(line)
        if depth > 0:
            continue
        text = '\n'.join(buf)
        buf, depth = [], 0
        try:
            rt.run(text)
        except OmarisError as e:
            print("Omaris error: %s" % e)


def main(argv):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    if len(argv) < 2:
        return repl()
    path = argv[1]
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            text = f.read()
    except OSError as e:
        print("Omaris error: cannot read %s: %s" % (path, e))
        return 1
    rt = Interp()
    try:
        rt.run(text)
    except OmarisError as e:
        print("Omaris error: %s" % e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
