# Waldemar

Waldemar is a compact UCI-compatible chess engine written in Go.

## Build (macOS)

```bash
go build -o waldemar .
```

Run:

```bash
./waldemar
```

## Cross-compile from macOS

### Windows (.exe)

```bash
GOOS=windows GOARCH=amd64 go build -o waldemar.exe .
```

### Linux

```bash
GOOS=linux GOARCH=amd64 go build -o waldemar-linux .
```

## UCI support

Implemented core UCI commands:

- `uci` → `id name Waldemar`, `id author ...`, `uciok`
- `isready` → `readyok`
- `ucinewgame`
- `position startpos` / `position fen ...` with optional `moves ...`
- `setoption name Threads value <n>` to select the fixed number of search workers
- `go infinite`, `go movetime <ms>`, `go depth <n>`, `go wtime <ms> btime <ms> winc <ms> binc <ms>`
- `stop`
- `quit`

Engine returns `bestmove <move>` and emits simple `info depth ...` lines during completed iterative search depths.