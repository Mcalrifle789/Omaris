# The Omaris Language Specification

**Version 0.1.0** — reference specification, derived from the original Omaris concept notes ("Omaris (Final prompt).DOCX" and the two handwritten concept pages).

Omaris is a general-purpose programming language built around one central idea: **certain instructions act as storage containers in their own right**, so a project's code and data never need to live on the developer's local disk, a flash drive, or an external hard drive. It is intended to be general-purpose enough to build large language models, image-generation models, code/chat models, audio/music models, and other AI model types, with its own functions, commands, and instruction set — a self-contained language, not a wrapper around an existing one.

This specification defines the language precisely. The reference implementation is `omaris.py` (a lexer → parser → tree-walking evaluator, written from scratch, standard library only).

---

## 1. Program structure

- Omaris programs are **line-based**. Each line is one instruction.
- Lines beginning with `#` are comments. Blank lines are ignored.
- **Blocks** are opened by a block instruction and closed by the keyword `end`.
- File extension: `.omr`.
- Unicode calculus symbols are native operators; UTF-8 source is expected.

## 2. Core instructions

### 2.1 Model lifecycle

| Instruction | Purpose |
|---|---|
| `model:create <Name>` | Begins creation of a new model. A named namespace whose inner functions are its lifecycle. |
| `def.fun..:(create)` | The creation function (training / setup phase of a model). |
| `def.fun..:(req)` | The formal request function (inference / serving phase of a model). |
| `tr.fun.:(dist.) <src> -> <dst>` | Starts distribution of items (pockets of stored data) to another system/container. |
| `tr.fun.:(dist.; value) <expr> -> <dst>` | Starts distribution of valued items — evaluates an expression and ships the value. |

Generic functions are declared the same way, with any name and parameters:

```
def.fun..:(<name>; <param>; <param>; ...)
  ...
  give <expr>          # return value
end
```

Calls: `name(arg, ...)`, or model lifecycle calls `Name.req(...)` / `Name.create(...)`.

### 2.2 Storage instructions

| Instruction | Purpose |
|---|---|
| `store.fun.:(open)` | Enables a storage instruction to run on a **local device**; mode applied to subsequently created containers. |
| `store.fun.:(closed)` | Enables a storage instruction to run on a **host**. |
| `store.fun.;start [name] [/capacity:...]` | Begins a storage function. **The instruction itself is the storage unit**: every line of code beneath it (until `end`) is absorbed into the container — effectively unlimited storage. |
| `store.fun.;status <name>` | Reports tracked weight, true size, pockets, and event log. |
| `store.fun.;run <name>` | Re-enters the runtime: executes the code captured inside the container. |

**Capacity modifiers** for `store.fun.;start`:

```
store.fun.;start <name> /capacity:GB(#)        # cap at # gigabytes
store.fun.;start <name> /capacity:TB(#)        # cap at # terabytes
store.fun.;start <name> /capacity:Infinite     # no cap (default)
```

The name may appear before or after the capacity modifier. Exceeding a declared capacity is a runtime error.

### 2.3 Hidden sub-storage: pockets

Code and data stored inside a `store.fun.;start` container move in and out of **hidden data pockets** — sub-storage regions nested inside the primary one.

```
pocket:add <container> <pocket>
pocket:put <container> <pocket> <key> = <expr>
pocket:move <container> <fromPocket> <toPocket> <key>
```

Shifting code/data between pockets triggers the same **reset-to-zero** on the tracked value at both ends, while the code's physical position does not change — only its tracked weight does.

### 2.4 The weight-reset semantics (faithful to the original concept)

Whenever lines of code enter the storage, or shift between pockets, or are re-run, the tracked value — the line count and the "weight" (size) of the accumulated code — **resets to 0, while the code still remains intact**. The container therefore always reads as lightweight no matter how much is stored underneath it.

Design honesty (per the concept notes): the reset does not remove data or reduce the machine's work — the code is explicitly still present. The interpreter therefore keeps a `true size` (lines + bytes) alongside the tracked weight, which resets to 0 on every entry/transfer event. As specified in the notes, if a genuine performance or storage benefit is wanted, the mechanism should be backed by compression, deduplication, lazy-loading, or offloading to another store; as built, the reset is a **display/abstraction feature** (hiding size/complexity from the user), exactly as the concept note framed the fallback. Every reset is journaled in the container's event log, visible via `store.fun.;status`.

## 3. Native calculus operators

Omaris expresses accumulation and rate-of-change directly in syntax — no library calls:

| Operator | Meaning | Over a list `[a₁, a₂, …]` |
|---|---|---|
| `∫` | Integral — cumulative totals, areas under curves, states over time for continuous/changing data | running total `[a₁, a₁+a₂, …]` |
| `∫∫` | Repeated integral | cumulative of cumulative |
| `∫∫∫` | Triple integral | cumulative of cumulative of cumulative |
| `ẏ` | First derivative — rate of change | `[a₂−a₁, a₃−a₂, …]` |
| `ÿ` | Second derivative — acceleration | difference of differences |

They are prefix operators and compose: `∫∫ series`, `ÿ(ẏ series)`. ASCII equivalents are available as builtins: `cumsum(x)`, `diff(x)`, `diff2(x)`.

## 4. Core language

Variables, expressions, control flow, and functions are orthogonal to the special instructions:

```
let name = <expr>              # assignment
out <expr>                     # print
if <cond> then ... else ... end
repeat <n> as <i> ... end      # counted loop (1..n)
repeat <list> as <item> ... end
repeat <n> ... end
give <expr>                    # return from a function
<expr>                         # bare expression (e.g. a function call)
```

Expressions: numbers, `"strings"`, lists `[1, 2, 3]`, dicts `{key: value}`,
`+ - * / % **`, comparisons `== != < <= > >=`, boolean `and or not`,
prefix `− ∫ ∫∫ ∫∫∫ ẏ ÿ`, calls `f(x)`, indexing `xs[0]`.

Builtins: `sum mean min max abs sqrt sin cos tan floor ceil round len range str num upper lower cumsum diff diff2`, plus constants `true false`.

## 5. Complete worked example

```
model:create WordGen
  def.fun..:(create; corpus)
    give corpus
  end

  def.fun..:(req; corpus; word)
    # bigram inference: most frequent word following `word`
    ...
    give best
  end
end

store.fun.:(open)
store.fun.;start training_corpus /capacity:TB(1)
tr.fun.:(dist.; value) ["omaris", "is", ...] -> training_corpus
end

let corpus = ["omaris", "is", "a", "language", ...]
out WordGen.req(corpus, "omaris")
```

See `examples/hello.omr`, `examples/storage.omr`, and `examples/model.omr` — all runnable with `python omaris.py examples/<file>.omr`.

## 6. Formal grammar (informal EBNF)

```
program    := { line } ;
line       := comment | instruction ;
instruction:= "let" NAME "=" expr | "out" expr | "give" [expr]
            | "if" expr "then" block [ "else" block ] "end"
            | "repeat" expr [ "as" NAME ] block "end"
            | "model:create" NAME block "end"
            | "def.fun" ".+" ":(" paramlist ")" block "end"
            | "store.fun" ".+" ":(open" | "closed")"
            | "store.fun" ".;" "start" [NAME] [ capacity ] block "end"
            | "store.fun" ".;" ("status"|"run") NAME
            | "tr.fun" ".+" ":(dist." | "dist.; value") expr "->" ref
            | "pocket" ":" ("add"|"put"|"move") ...
            | expr ;
capacity   := "/capacity:" ( "GB" "(" NUM ")" | "TB" "(" NUM ")" | "Infinite" ) ;
ref        := NAME [ "." NAME ] ;     (* container [. pocket] *)
expr       := calc ops with precedence or > and > not > cmp > +- > */% >
              unary(− ∫ ∫∫ ∫∫∫ ẏ ÿ) > ** > postfix(call/index) > atom ;
```

## 7. Extended Edition — function starters and domains

*(from the Omaris Extended Edition concept document and handwritten pages)*

Omaris is a **core foundation language** for building applications, websites, AI agents, audio systems, and LLMs. Every extended command begins with a **function starter** that routes it to its domain:

| Starter | Domain |
|---|---|
| `fun.` | Function starter — used for most of the language, since most of it deals with code functions |
| `deb.` | Type of function starter used for agentic code commands |
| `aud.` | Type of function code starter — prioritizes converting code into sound |
| `edm.` | Function code starter for LLM creation: coding, stabilization, and fine-tuning of the LLM it is creating |
| `oma.` | Prioritizes website or app building, as well as other project builds |
| `fun.ag` | Prioritized in building AI agents and other bot agents |

### 7.1 `fun.` — text rendering

```
fun.text/print("Hello")/(2000 × 1500)      # prints "Hello" placed at that point on the page
text.set-style/(STYLE)                     # sets a text style (bold, underline, red, ...)
text.rainbowpalette/animation              # arms the rainbow palette
text.animation/set animation               # + this = animated rainbow text
```

### 7.2 `edm.` — LLM & code control

```
edm.pro.:edit <ref>          # turns a website, app, or other coded build into a functional sandbox
edm.:next/transition <src> -> <dst>   # moves code to another location (functions into containers, pockets between containers)
edm.:next/stop <function>    # stops the code being live — disables that code
edm.:next/resume <function>  # resumes disabled code back into live code
```

### 7.3 `oma.` — app, website & project building

```
oma.:appbuild/start <name>          # starts the building of an application
oma.:appbuild/start. <a> <b> ...    # starts the building of multiple apps (trailing dot)
oma.://app/next [name]              # moves on to the next part of the app building operation
oma.://app/stop [name]              # stops the app building operation
oma://app/pause [name]              # pauses the app building process
oma.://app/cache [name]             # caches the entire app (into an Omaris storage container)
oma.://webbuild/start <site>        # starts website coding operations
oma.://web/start <site>             # starts website coding operations
oma.://web/auth <source>            # sends requests to authenticate sources used for website coding
oma.://web/next [name]              # moves on to the next coding operation
oma.://web/end [name]               # ends the website coding operation
oma.://web/format;start [site]      # starts formatting a wireframe and database for a site
oma.://web/design-format;start [site]  # starts a design operation
oma.://web/design; find-design-skill <query>   # finds a design skill on the internet
oma.://internet;search/start <query>   # starts an internet search on the user's browser
oma.://internet;search/end             # ends an internet search on the user's browser
```

### 7.4 `oma.` — music & audio

Audio buffers are created from code with the `aud.` starter and shaped by the `oma.` music commands:

```
aud.:tone <name>/(freq; ms)          # converts a value into sound (a real PCM tone buffer)
oma.://music;audio//(combine-audio-file) <a> <b> [-> <out>]   # combines two or more audio files
oma.://music;audio//(enhance) <name>      # enhances the audio file(s)
oma.://music;audio/(set-loop) <name> <n>  # sets audio loop
oma.://music;audio//(reverb) <name> = SET INTENSITY <n>   # applies reverb to an audio file (or files)
aud.:play <name> ["file.wav"]        # renders the buffer to a real .wav file
```

### 7.5 `fun.ag` / `deb.` — AI agents & automation

```
fun.ag.://agent <name>               # creates an agent
fun.ag.://agentflow. <name> ... end  # begins a workflow with an agent
fun.ag.://agentflow;x = VALUE        # increases the power of the agent
fun.ag.://agentflow;work-speed-x = VALUE   # increases the work speed of the automation agent
fun.ag.://agent;start <name>         # starts an agent operation (runs its workflow)
```

**NOTICE (enforced by the runtime): the work speed value must be less than the power value.** A `work-speed-x` greater than or equal to `x` is a runtime error.

## 8. Design notes and future directions

- **`store.fun.;start` is the storage unit**, per the concept. A future native runtime would persist container state to a distributed host ("closed" mode) so code and data never live on the developer's local disk — the current reference interpreter models the semantics in-process.
- **Distribution** (`tr.fun.`) is the transport layer between systems: containers, pockets, and valued items.
- **Model lifecycle** (`model:create` / `def.fun..:(create)` / `def.fun..:(req)`) is the AI-shaped abstraction: everything needed to train and serve LLMs, image, code/chat, and audio/music models.
- **Calculus** is native so physical/continuous systems express accumulation and rates of change directly.
