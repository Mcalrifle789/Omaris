# Omaris

A programming language where **instructions are storage containers** — code and data are absorbed into the language's own storage instructions instead of living on local disk, flash drives, or external hard drives. Designed for building AI models (LLMs, image, code/chat, audio/music models) with a native model lifecycle and native calculus operators.

Built from scratch: `omaris.py` is a complete lexer → parser → tree-walking evaluator, standard library only. No host-language tricks, no wrapping.

## Quick start

```
python omaris.py examples\hello.omr
python omaris.py examples\storage.omr
python omaris.py examples\model.omr
python omaris.py examples\extended.omr      # Extended Edition demo
python omaris.py                # interactive REPL
```

## The idea in 30 seconds

```
store.fun.:(open)

store.fun.;start archive /capacity:GB(1)
  let note = "these lines are absorbed into the container itself"
  out "running code that lives inside storage"
end

store.fun.;status archive       # tracked weight: 0 — true size retained
pocket:add archive cold         # hidden sub-storage pocket
pocket:put archive cold gem = "secret payload"
tr.fun.:(dist.; value) "item for another system" -> archive
store.fun.;run archive          # re-enter the runtime from inside the container
```

Every time code enters the container, shifts between hidden pockets, or is re-run, the **tracked weight resets to 0 while the code remains intact** — the container always reads as lightweight, exactly as the concept specifies.

## Feature map

- **Model lifecycle** — `model:create`, `def.fun..:(create)`, `def.fun..:(req)`, distribution via `tr.fun.:(dist.)` / `tr.fun.:(dist.; value)`
- **Storage instructions** — `store.fun.:(open)` (local device) / `store.fun.:(closed)` (host); `store.fun.;start` blocks are the containers, with `/capacity:GB(#)` / `TB(#)` / `Infinite`
- **Hidden pockets** — sub-storage inside a container: `pocket:add` / `pocket:put` / `pocket:move`
- **Native calculus** — integrals `∫ ∫∫ ∫∫∫` (cumulative totals, areas under curves) and derivatives `ẏ ÿ` (rates of change), directly in syntax
- **General-purpose core** — variables, expressions, `if/then/else`, `repeat`, functions (`def.fun..:(name; params)`), lists, dicts, string/number builtins

### Extended Edition (core foundation)

- **Six function starters** — `fun.` (code), `deb.` (agentic), `aud.` (code→sound), `edm.` (LLM control), `oma.` (apps/websites), `fun.ag` (AI agents)
- **Text rendering** — `fun.text/print("Hello")/(2000 × 1500)`, `text.set-style/(STYLE)`, animated rainbow text
- **EDM control** — `edm.pro.:edit` (functional sandbox), `edm.:next/transition`, `edm.:next/stop` / `resume` (live-code control)
- **App & web building** — `oma.:appbuild/start`, app phases (next/stop/pause/cache), website pipeline (auth, wireframe/database formatting, design skills, internet search)
- **Music & audio** — `aud.:tone` creates real PCM buffers; combine / enhance / set-loop / reverb (= SET INTENSITY); `aud.:play` renders playable `.wav` files
- **AI agents** — `fun.ag.://agent`, agentflows, power (`x`) and work-speed tuning, with the enforced rule: *work speed must be less than power* (see [SPEC.md](SPEC.md) §7)

## Files

| File | What it is |
|---|---|
| `omaris.py` | The complete reference interpreter + REPL |
| `SPEC.md` | The full language specification |
| `examples/*.omr` | Runnable example programs |

Full details in [SPEC.md](SPEC.md).
