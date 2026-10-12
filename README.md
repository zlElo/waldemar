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
- `go infinite`, `go movetime <ms>`, `go depth <n>`, `go wtime <ms> btime <ms>`
- `stop`
- `quit`

Engine returns `bestmove <move>` and emits simple `info depth ...` lines during completed iterative search depths.

## Architecture (brief)

- **Move generation**: Full legal move generation from pseudo-legal moves, including castling, en passant, and promotions.
- **Search**: Iterative deepening alpha-beta (negamax style), with depth/time controls and cancellation support.
- **Evaluation**: Material balance plus piece-square tables.
- **Game state**: FEN parser, incremental make/undo move logic, check/checkmate/stalemate handling.

## Quick CLI check

```bash
printf 'uci\nisready\nposition startpos\ngo depth 3\nquit\n' | ./waldemar
```

## GUI usage

Add the built binary to any UCI GUI (Cute Chess, Arena, etc.) as a new UCI engine.
