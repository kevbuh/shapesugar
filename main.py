#!/usr/bin/env python3
# shapescript.py
# Shape-only DSL with:
# - multiplicative dim expressions (a*b*32)
# - per-call parameters (no global binding of 'in', 'out', etc.)
# - pattern variables are LOCAL per op (no leaking across ops)
# - multi-line pipelines (lines starting with `|>` continue)
# - inline comments supported anywhere (everything after '#' is ignored)
# No external deps. Python 3.9+

from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

# ---------------------------
# Token / identifier helpers
# ---------------------------

DIM_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|\d+|_|\*|(?:[A-Za-z_][A-Za-z0-9_]*|\d+)(?:\*(?:[A-Za-z_][A-Za-z0-9_]*|\d+))*)\s*")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
INT = re.compile(r"\d+$")

def strip_inline_comment(s: str) -> str:
    return s.split("#", 1)[0].rstrip()

def split_dims(s: str) -> List[str]:
    s = strip_inline_comment(s.strip())
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Expected [dims], got: {s}")
    inner = s[1:-1].strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    for p in parts:
        if not DIM_TOKEN.fullmatch(p):
            raise ValueError(f"Bad dim token: {p}")
    return parts

def is_int(tok: str) -> bool:
    return bool(INT.match(tok))

def is_ident(tok: str) -> bool:
    return bool(IDENT.match(tok))

def is_mul_expr(tok: str) -> bool:
    return ("*" in tok) and all(t and (is_ident(t) or is_int(t)) for t in tok.split("*"))

# ---------------------------
# Core data structures
# ---------------------------

@dataclass
class OpDef:
    name: str
    in_pat: List[str]
    out_pat: List[str]

@dataclass
class TensorDef:
    name: str
    shape: List[str]

class ShapeEnv:
    def __init__(self, init: Optional[Dict[str, Union[int, str]]] = None):
        self.bindings: Dict[str, Union[int, str]] = dict(init or {})

    def copy(self) -> "ShapeEnv":
        return ShapeEnv(dict(self.bindings))

    def bind_equal(self, a: str, b: Union[int, str]) -> None:
        if isinstance(b, str) and b in self.bindings:
            b = self.bindings[b]
        if is_int(a):
            ai = int(a)
            if isinstance(b, int):
                if ai != b:
                    raise ValueError(f"Mismatch: {ai} != {b}")
            elif isinstance(b, str):
                self.bind_dim(b, ai)
            else:
                raise ValueError("Internal bind error.")
            return
        if a in self.bindings:
            cur = self.bindings[a]
            if isinstance(cur, int) and isinstance(b, int):
                if cur != b: raise ValueError(f"Mismatch: {a} ({cur}) != {b}")
            elif isinstance(cur, str) and isinstance(b, str):
                if cur != b: raise ValueError(f"Mismatch: {a} ({cur}) != {b}")
            elif isinstance(cur, int) and isinstance(b, str):
                if b in self.bindings:
                    other = self.bindings[b]
                    if isinstance(other, int) and other != cur:
                        raise ValueError(f"Mismatch: {b} ({other}) != {cur}")
                    if isinstance(other, str) and other != a:
                        raise ValueError(f"Mismatch: {b} ({other}) != {a}")
                else:
                    self.bind_dim(b, cur)
            elif isinstance(cur, str) and isinstance(b, int):
                if cur in self.bindings:
                    cc = self.bindings[cur]
                    if isinstance(cc, int) and cc != b:
                        raise ValueError(f"Mismatch: {cur} ({cc}) != {b}")
                else:
                    self.bind_dim(cur, b)
        else:
            self.bind_dim(a, b)

    def bind_dim(self, sym: str, value: Union[int, str]) -> None:
        if is_int(sym):
            raise ValueError("Attempted to bind into a literal int.")
        if sym in self.bindings:
            cur = self.bindings[sym]
            if isinstance(cur, int) and isinstance(value, int):
                if cur != value:
                    raise ValueError(f"Mismatch: {sym} already {cur}, tried {value}")
            elif isinstance(cur, str) and isinstance(value, str):
                if cur != value:
                    raise ValueError(f"Mismatch: {sym} already {cur}, tried {value}")
            elif isinstance(cur, int) and isinstance(value, str):
                if value in self.bindings:
                    ov = self.bindings[value]
                    if isinstance(ov, int) and ov != cur:
                        raise ValueError(f"Mismatch: {value} already {ov}, != {cur}")
                    if isinstance(ov, str) and ov != sym:
                        raise ValueError(f"Mismatch: {value} already {ov}, != {sym}")
                else:
                    self.bindings[value] = cur
            elif isinstance(cur, str) and isinstance(value, int):
                if cur in self.bindings:
                    ov = self.bindings[cur]
                    if isinstance(ov, int) and ov != value:
                        raise ValueError(f"Mismatch: {cur} already {ov}, != {value}")
                else:
                    self.bindings[cur] = value
        else:
            self.bindings[sym] = value

    def resolve_symbol(self, tok: str) -> Union[int, str]:
        if is_int(tok): 
            return int(tok)
        return self.bindings.get(tok, tok)

    def eval_dim_expr(self, tok: str) -> Union[int, str]:
        tok = tok.strip()
        if tok in ("*", "_"):
            return tok
        if is_int(tok):
            return int(tok)
        if is_ident(tok):
            return self.resolve_symbol(tok)
        if is_mul_expr(tok):
            prod: Optional[int] = 1  # type: ignore
            unresolved: List[str] = []
            for part in tok.split("*"):
                if is_int(part):
                    prod *= int(part)  # type: ignore
                else:
                    val = self.resolve_symbol(part)
                    if isinstance(val, int):
                        prod *= val  # type: ignore
                    else:
                        unresolved.append(str(val))
            if unresolved:
                raise ValueError(f"Unbound symbols in expression '{tok}': {', '.join(unresolved)}")
            return int(prod)  # type: ignore
        raise ValueError(f"Bad dim token: {tok}")

    def __repr__(self) -> str:
        return f"ShapeEnv({self.bindings})"

# ---------------------------
# Matching & applying ops
# ---------------------------

def match_pattern(in_shape: List[Union[int,str]],
                  pat: List[str],
                  env: ShapeEnv) -> Tuple[List[Union[int,str]], Dict[str, Union[int,str]]]:
    pack_index = None
    if "*" in pat:
        pack_index = pat.index("*")

    if pack_index is None and len(in_shape) != len(pat):
        raise ValueError(f"Arity mismatch: got {len(in_shape)} dims, expected {len(pat)}")
    if pack_index is not None and len(in_shape) < (len(pat) - 1):
        raise ValueError(f"Arity mismatch: got {len(in_shape)} dims, expected at least {len(pat)-1}")

    updates: Dict[str, Union[int,str]] = {}

    def bind(a: str, b: Union[int,str]):
        if a == "_": return
        env.bind_equal(a, b)
        updates[a] = env.resolve_symbol(a)

    if pack_index is None:
        for a, b in zip(pat, in_shape):
            if a != "_":
                bind(a, b)
        captured = []
    else:
        left_pat = pat[:pack_index]
        right_pat = pat[pack_index+1:]
        L = len(left_pat); R = len(right_pat)
        left_in = in_shape[:L]
        right_in = in_shape[-R:] if R > 0 else []
        pack_in = in_shape[L:len(in_shape)-R] if R > 0 else in_shape[L:]

        for a, b in zip(left_pat, left_in):
            if a != "_": bind(a, b)
        for a, b in zip(right_pat, right_in):
            if a != "_": bind(a, b)
        captured = list(pack_in)
    return captured, updates

def build_output(out_pat: List[str],
                 env: ShapeEnv,
                 captured_pack: List[Union[int,str]]) -> List[Union[int,str]]:
    result: List[Union[int,str]] = []
    for tok in out_pat:
        if tok == "*":
            result.extend(captured_pack)
        elif tok == "_":
            result.append("_")
        else:
            result.append(env.resolve_symbol(tok))
    return result

# ---------------------------
# Parser for the DSL
# ---------------------------

OP_DEF_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\[[^\]]*\])\s*->\s*(\[[^\]]*\])\s*$")
TENSOR_DEF_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\[[^\]]*\])\s*$")
LET_RE = re.compile(r"^\s*let\s+(.+)$")
PIPE_START_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=\s*)?([A-Za-z_][A-Za-z0-9_]*)?\s*\|\>\s*(.+)$")
PIPE_CONT_RE = re.compile(r"^\s*\|\>\s*(.+)$")
CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\((.*?)\))?\s*$")

def parse_kv_args(s: str) -> Dict[str, str]:
    s = strip_inline_comment(s.strip())
    if not s:
        return {}
    pairs = [p.strip() for p in s.split(",") if p.strip()]
    res: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Expected key=value in call args, got '{p}'")
        k, v = [x.strip() for x in p.split("=", 1)]
        if not IDENT.fullmatch(k):
            raise ValueError(f"Bad arg name '{k}'")
        v = strip_inline_comment(v)
        if not DIM_TOKEN.fullmatch(v):
            raise ValueError(f"Bad arg value '{v}' (use ints, identifiers, or a*b*c)")
        res[k] = v
    return res

class ShapeScript:
    def __init__(self):
        self.ops: Dict[str, OpDef] = {}
        self.tensors: Dict[str, TensorDef] = {}
        self.env = ShapeEnv()

    def add_op(self, name: str, in_pat: List[str], out_pat: List[str]):
        if name in self.ops:
            prev = self.ops[name]
            if prev.in_pat == in_pat and prev.out_pat == out_pat:
                return
        bad_pat = [t for t in in_pat + out_pat if is_mul_expr(t)]
        if bad_pat:
            raise ValueError(f"Op patterns cannot contain multiplicative expressions: {bad_pat}")
        self.ops[name] = OpDef(name, in_pat, out_pat)

    def add_tensor(self, name: str, shape: List[str]):
        self.tensors[name] = TensorDef(name, shape)

    def set_lets(self, kvs: Dict[str, Union[int,str]]):
        for k, v in kvs.items():
            self.env.bind_dim(k, v)

    def _subst_pattern_with_call_args(self, pat: List[str], call_args: Dict[str, Union[int, str]]) -> List[str]:
        out: List[str] = []
        for tok in pat:
            if tok in call_args:
                val = call_args[tok]
                out.append(str(val) if isinstance(val, int) else val)
            else:
                out.append(tok)
        return out

    def apply_op(self, op: OpDef, in_shape: List[Union[int,str]], call_args_raw: Dict[str, str]) -> List[Union[int,str]]:
        # Local working env (copy of globals)
        local = self.env.copy()

        # Evaluate call args in local env, but DO NOT bind their names globally
        call_args: Dict[str, Union[int, str]] = {}
        for k, raw in call_args_raw.items():
            if raw in ("*", "_"):
                raise ValueError(f"Call argument '{k}' cannot be '*' or '_'")
            raw = strip_inline_comment(raw)
            val = local.eval_dim_expr(raw)
            if isinstance(val, str) and not is_ident(val):
                raise ValueError(f"In call to {op.name}, arg {k} must resolve to an int or identifier; got '{val}'")
            call_args[k] = val

        # Substitute call-arg names into patterns (per-call substitution)
        in_pat_sub  = self._subst_pattern_with_call_args(op.in_pat,  call_args)
        out_pat_sub = self._subst_pattern_with_call_args(op.out_pat, call_args)

        # Match & build output using ONLY local env
        captured, _ = match_pattern(in_shape, in_pat_sub, local)
        out = build_output(out_pat_sub, local, captured)

        # IMPORTANT: Do NOT overwrite global env with local; pattern symbols are per-call.
        # Keep only the original global env (lets etc.).
        # If you ever want selective propagation, implement a whitelist here.

        return out

    def run_pipeline(self, start_tensor: str, pipeline: List[Tuple[str, Dict[str, str]]]) -> List[Union[int,str]]:
        if start_tensor not in self.tensors:
            raise ValueError(f"Unknown tensor '{start_tensor}'")
        cur: List[Union[int,str]] = []
        for tok in self.tensors[start_tensor].shape:
            cur.append(self.env.eval_dim_expr(tok))
        print(f"{start_tensor}: {pretty_shape(cur)}")
        for (opname, args_raw) in pipeline:
            if opname not in self.ops:
                raise ValueError(f"Unknown op '{opname}'")
            op = self.ops[opname]
            before = pretty_shape(cur)
            cur = self.apply_op(op, cur, args_raw)
            after = pretty_shape(cur)
            arg_txt = "" if not args_raw else " " + ", ".join(f"{k}={args_raw[k]}" for k in args_raw)
            print(f"  |> {opname}{arg_txt}: {before} -> {after}")
        return cur

def pretty_dim(d: Union[int,str]) -> str:
    return str(d)

def pretty_shape(sh: List[Union[int,str]]) -> str:
    return "[" + ", ".join(pretty_dim(d) for d in sh) + "]"

def _parse_stage_text(stage_text: str, line_no: int) -> Tuple[str, Dict[str, str]]:
    stage_text = strip_inline_comment(stage_text)
    cm = CALL_RE.match(stage_text.strip())
    if not cm:
        raise ValueError(f"[line {line_no}] Bad pipeline stage: '{stage_text.strip()}'")
    opname = cm.group(1)
    args_str = cm.group(3) or ""
    args = parse_kv_args(args_str)
    return opname, args

def parse_program(text: str) -> ShapeScript:
    ss = ShapeScript()
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    current_start: Optional[str] = None
    current_stages: List[Tuple[str, Dict[str, str]]] = []

    def flush_pipeline():
        nonlocal current_start, current_stages
        if current_start is not None:
            ss.run_pipeline(current_start, current_stages)
            current_start = None
            current_stages = []

    for i, raw in enumerate(lines, start=1):
        raw_no_comment = strip_inline_comment(raw)
        ln = raw_no_comment.strip()

        if not ln:
            flush_pipeline()
            continue

        mcont = PIPE_CONT_RE.match(raw_no_comment)
        if mcont and current_start is not None:
            stage_text = mcont.group(1)
            current_stages.append(_parse_stage_text(stage_text, i))
            continue
        elif mcont and current_start is None:
            raise ValueError(f"[line {i}] Pipeline continuation without a start")

        mstart = PIPE_START_RE.match(raw_no_comment)
        if mstart:
            flush_pipeline()
            start_tensor = mstart.group(1) if mstart.group(3) is None else mstart.group(3)
            tail = mstart.group(4)
            stages = ["|>" + s for s in tail.split("|>") if s.strip()]
            current_start = start_tensor
            current_stages = []
            for st in stages:
                stage_text = st.replace("|>", "", 1)
                current_stages.append(_parse_stage_text(stage_text, i))
            continue

        # let bindings
        m = LET_RE.match(ln)
        if m:
            kvs = {}
            body = m.group(1)
            pairs = [p.strip() for p in body.split(",")]
            for p in pairs:
                if not p: continue
                if "=" not in p:
                    raise ValueError(f"[line {i}] Expected key=value in let, got '{p}'")
                k, v = [x.strip() for x in p.split("=", 1)]
                if not IDENT.fullmatch(k):
                    raise ValueError(f"[line {i}] Bad let name '{k}'")
                if is_int(v):
                    kvs[k] = int(v)
                elif IDENT.fullmatch(v):
                    kvs[k] = v
                else:
                    raise ValueError(f"[line {i}] Bad let value '{v}' (use ints or identifiers)")
            ss.set_lets(kvs)
            continue

        # op def
        m = OP_DEF_RE.match(ln)
        if m:
            name, inp, outp = m.groups()
            in_pat = split_dims(inp)
            out_pat = split_dims(outp)
            ss.add_op(name, in_pat, out_pat)
            continue

        # tensor def
        m = TENSOR_DEF_RE.match(ln)
        if m:
            name, shp = m.groups()
            ss.add_tensor(name, split_dims(shp))
            continue

        raise ValueError(f"[line {i}] Could not parse: {raw}")

    flush_pipeline()
    return ss

# ---------------------------
# A small standard library
# ---------------------------

STD_LIB = """
relu: [*] -> [*]
gelu: [*] -> [*]
tanh: [*] -> [*]
sigmoid: [*] -> [*]

linear: [*, in] -> [*, out]
softmax: [*] -> [*]
layernorm: [*] -> [*]

# Use capital H/W to avoid clashing with user 'h', 'w'
conv2d: [b, ch, H, W] -> [b, ch_out, H_out, W_out]

matmul: [*, m, k] -> [*, m, n]

# You assert the flattened 'dim' size (batch excluded).
flatten_to: [*, _] -> [*, dim]

reshape: [*] -> [*]
concat: [*, d] -> [*, d]
"""

# ---------------------------
# Demo / CLI
# ---------------------------

DEMO = r"""
let b=32, d=784, h=256, c=10, heads=8, q=64, t=128

X: [b, d]
IMG: [b, 3, 28, 28]
TOK: [b, t, heads*q]

# MLP
X |> linear(in=d, out=h) |> relu |> linear(in=h, out=c) |> softmax

# CNN
IMG |> conv2d(ch_out=16, H_out=26, W_out=26)
    |> relu
    |> conv2d(ch_out=32, H_out=24, W_out=24)
    |> relu
    |> flatten_to(dim=32*24*24)
    |> linear(in=32*24*24, out=128)
    |> relu
    |> linear(in=128, out=c)
    |> softmax

# Attention-ish
TOK |> linear(in=heads*q, out=h) |> relu |> linear(in=h, out=heads*q)
"""

def main():
    text = (STD_LIB + "\n" + DEMO) if len(sys.argv) == 1 else open(sys.argv[1], "r").read()
    try:
        parse_program(text)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
