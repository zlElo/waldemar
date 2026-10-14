"""
converter.py — converts a PGN (including variations/RAV) into the
WBF book format used by the "Waldemar" Go engine:

<zobrist-hash-hex> <uci-move> <weight>

The Zobrist hash is calculated exactly as it is in the engine
(initZobrist/computeHash): using the same Xorshift PRNG, the same
Zobrist keys, and the same hash composition. The resulting file
can be loaded directly by the engine using LoadBook().

Usage:
python3 converter.py input.pgn book.wbf [options]

Options:
--max-ply N      Maximum number of plies (half-moves) to record (default: 20)
--mainlines      Include only main lines; variations are ignored

Uses only the standard library; no external packages required.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field

WHITE, BLACK = 1, -1
EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5, 6

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8


def _init_zobrist():
    x = 0x9E3779B97F4A7C15
    M = (1 << 64) - 1

    def nxt():
        nonlocal x
        x ^= x >> 12
        x ^= (x << 25) & M
        x ^= x >> 27
        return (x * 2685821657736338717) & M

    piece = [[nxt() for _ in range(64)] for _ in range(12)]
    castling = [nxt() for _ in range(16)]
    ep_file = [nxt() for _ in range(9)]
    side = nxt()
    return piece, castling, ep_file, side


Z_PIECE, Z_CASTLE, Z_EP, Z_SIDE = _init_zobrist()


def piece_hash_idx(piece):
    if piece > 0:
        return piece - 1
    return 6 + (-piece - 1)


def ep_file_idx(ep):
    if ep < 0 or ep >= 64:
        return 0
    return (ep % 8) + 1


FEN_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

FEN_PIECE = {"P": PAWN, "N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN, "K": KING,
             "p": -PAWN, "n": -KNIGHT, "b": -BISHOP, "r": -ROOK, "q": -QUEEN, "k": -KING}

CASTLE_MASK = [15] * 64
CASTLE_MASK[0] = 15 ^ CASTLE_WQ
CASTLE_MASK[7] = 15 ^ CASTLE_WK
CASTLE_MASK[4] = 15 ^ (CASTLE_WK | CASTLE_WQ)
CASTLE_MASK[56] = 15 ^ CASTLE_BQ
CASTLE_MASK[63] = 15 ^ CASTLE_BK
CASTLE_MASK[60] = 15 ^ (CASTLE_BK | CASTLE_BQ)


@dataclass
class Position:
    board: list = field(default_factory=lambda: [0] * 64)
    side: int = WHITE
    castling: int = 0
    ep: int = -1

    @staticmethod
    def from_fen(fen):
        p = Position()
        parts = fen.split()
        rows = parts[0].split("/")
        for r in range(8):
            f = 0
            for ch in rows[r]:
                if ch.isdigit():
                    f += int(ch)
                else:
                    p.board[(7 - r) * 8 + f] = FEN_PIECE[ch]
                    f += 1
        p.side = WHITE if parts[1] == "w" else BLACK
        if len(parts) > 2 and parts[2] != "-":
            if "K" in parts[2]: p.castling |= CASTLE_WK
            if "Q" in parts[2]: p.castling |= CASTLE_WQ
            if "k" in parts[2]: p.castling |= CASTLE_BK
            if "q" in parts[2]: p.castling |= CASTLE_BQ
        if len(parts) > 3 and parts[3] != "-":
            p.ep = (int(parts[3][1]) - 1) * 8 + (ord(parts[3][0]) - 97)
        return p

    def copy(self):
        return Position(self.board[:], self.side, self.castling, self.ep)

    def hash(self):
        h = 0
        for sq, pc in enumerate(self.board):
            if pc != EMPTY:
                h ^= Z_PIECE[piece_hash_idx(pc)][sq]
        h ^= Z_CASTLE[self.castling] ^ Z_EP[ep_file_idx(self.ep)]
        if self.side == BLACK:
            h ^= Z_SIDE
        return h

    def king_sq(self, side):
        target = KING * side
        for sq in range(64):
            if self.board[sq] == target:
                return sq
        return -1

    def attacked(self, sq, by):
        for df, dr in ((1, 2), (2, 1), (-1, 2), (-2, 1),
                       (1, -2), (2, -1), (-1, -2), (-2, -1)):
            f, r = sq % 8 + df, sq // 8 + dr
            if 0 <= f < 8 and 0 <= r < 8 and self.board[r * 8 + f] == KNIGHT * by:
                return True
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                f, r = sq % 8 + df, sq // 8 + dr
                if 0 <= f < 8 and 0 <= r < 8 and self.board[r * 8 + f] == KING * by:
                    return True
        dr = -1 if by == WHITE else 1
        for df in (-1, 1):
            f, r = sq % 8 + df, sq // 8 + dr
            if 0 <= f < 8 and 0 <= r < 8 and self.board[r * 8 + f] == PAWN * by:
                return True
        for dirs, first_type in (
            (((1, 1), (1, -1), (-1, 1), (-1, -1)), BISHOP),
            (((1, 0), (-1, 0), (0, 1), (0, -1)), ROOK),
        ):
            for df, dr in dirs:
                f, r = sq % 8 + df, sq // 8 + dr
                while 0 <= f < 8 and 0 <= r < 8:
                    pc = self.board[r * 8 + f]
                    if pc != EMPTY:
                        if pc in (first_type * by, QUEEN * by):
                            return True
                        break
                    f += df
                    r += dr
        return False

    def in_check(self, side):
        ks = self.king_sq(side)
        return ks >= 0 and self.attacked(ks, -side)

    def pseudo_moves(self):
        out = []
        side = self.side
        for sq in range(64):
            pc = self.board[sq]
            if pc == EMPTY or (pc > 0) != (side > 0):
                continue
            pt = abs(pc)
            f, r = sq % 8, sq // 8
            if pt == PAWN:
                dr = 1 if side == WHITE else -1
                promo_rank = 6 if side == WHITE else 1
                start_rank = 1 if side == WHITE else 6
                one = sq + 8 * dr
                if 0 <= one < 64 and self.board[one] == EMPTY:
                    if r == promo_rank:
                        for pr in (QUEEN, ROOK, BISHOP, KNIGHT):
                            out.append((sq, one, pr, False))
                    else:
                        out.append((sq, one, 0, False))
                        if r == start_rank and self.board[one + 8 * dr] == EMPTY:
                            out.append((sq, one + 8 * dr, 0, False))
                for df in (-1, 1):
                    tf, tr = f + df, r + dr
                    if not (0 <= tf < 8 and 0 <= tr < 8):
                        continue
                    to = tr * 8 + tf
                    tgt = self.board[to]
                    if tgt != EMPTY and (tgt > 0) != (side > 0):
                        if r == promo_rank:
                            for pr in (QUEEN, ROOK, BISHOP, KNIGHT):
                                out.append((sq, to, pr, False))
                        else:
                            out.append((sq, to, 0, False))
                    elif to == self.ep:
                        out.append((sq, to, 0, True))
            elif pt == KNIGHT:
                for df, dr in ((1, 2), (2, 1), (-1, 2), (-2, 1),
                               (1, -2), (2, -1), (-1, -2), (-2, -1)):
                    tf, tr = f + df, r + dr
                    if 0 <= tf < 8 and 0 <= tr < 8:
                        tgt = self.board[tr * 8 + tf]
                        if tgt == EMPTY or (tgt > 0) != (side > 0):
                            out.append((sq, tr * 8 + tf, 0, False))
            elif pt == KING:
                for df in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        if df == 0 and dr == 0:
                            continue
                        tf, tr = f + df, r + dr
                        if 0 <= tf < 8 and 0 <= tr < 8:
                            tgt = self.board[tr * 8 + tf]
                            if tgt == EMPTY or (tgt > 0) != (side > 0):
                                out.append((sq, tr * 8 + tf, 0, False))
            else:
                dirs = []
                if pt in (BISHOP, QUEEN):
                    dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                if pt in (ROOK, QUEEN):
                    dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for df, dr in dirs:
                    tf, tr = f + df, r + dr
                    while 0 <= tf < 8 and 0 <= tr < 8:
                        to = tr * 8 + tf
                        tgt = self.board[to]
                        if tgt == EMPTY:
                            out.append((sq, to, 0, False))
                        else:
                            if (tgt > 0) != (side > 0):
                                out.append((sq, to, 0, False))
                            break
                        tf += df
                        tr += dr
        if side == WHITE and self.board[4] == KING:
            if (self.castling & CASTLE_WK and self.board[5] == 0 and self.board[6] == 0
                    and not self.attacked(4, BLACK) and not self.attacked(5, BLACK)
                    and not self.attacked(6, BLACK)):
                out.append((4, 6, 0, False))
            if (self.castling & CASTLE_WQ and self.board[3] == 0 and self.board[2] == 0
                    and self.board[1] == 0
                    and not self.attacked(4, BLACK) and not self.attacked(3, BLACK)
                    and not self.attacked(2, BLACK)):
                out.append((4, 2, 0, False))
        if side == BLACK and self.board[60] == -KING:
            if (self.castling & CASTLE_BK and self.board[61] == 0 and self.board[62] == 0
                    and not self.attacked(60, WHITE) and not self.attacked(61, WHITE)
                    and not self.attacked(62, WHITE)):
                out.append((60, 62, 0, False))
            if (self.castling & CASTLE_BQ and self.board[59] == 0 and self.board[58] == 0
                    and self.board[57] == 0
                    and not self.attacked(60, WHITE) and not self.attacked(59, WHITE)
                    and not self.attacked(58, WHITE)):
                out.append((60, 58, 0, False))
        return out

    def make(self, mv):
        frm, to, promo, is_ep = mv
        side = self.side
        piece = self.board[frm]
        cap_sq = to - 8 if (is_ep and side == WHITE) else (to + 8 if is_ep else to)
        self.board[frm] = EMPTY
        if is_ep:
            self.board[cap_sq] = EMPTY
        self.board[to] = (promo if side == WHITE else -promo) if promo else piece
        if abs(piece) == KING and abs(to - frm) == 2:
            if to == 6:
                self.board[5], self.board[7] = self.board[7], EMPTY
            elif to == 2:
                self.board[3], self.board[0] = self.board[0], EMPTY
            elif to == 62:
                self.board[61], self.board[63] = self.board[63], EMPTY
            elif to == 58:
                self.board[59], self.board[56] = self.board[56], EMPTY
        self.castling &= CASTLE_MASK[frm] & CASTLE_MASK[to]
        self.ep = frm + (8 if side == WHITE else -8) \
            if (abs(piece) == PAWN and abs(to - frm) == 16) else -1
        self.side = -side

    def legal_moves(self):
        side = self.side
        out = []
        for mv in self.pseudo_moves():
            cp = self.copy()
            cp.make(mv)
            if not cp.in_check(side):
                out.append(mv)
        return out

    def is_castle(self, mv):
        return abs(self.board[mv[0]]) == KING and abs(mv[1] - mv[0]) == 2


def sq_name(sq):
    return chr(97 + sq % 8) + str(sq // 8 + 1)


PROMO_CH = {QUEEN: "q", ROOK: "r", BISHOP: "b", KNIGHT: "n"}


def mv_uci(mv):
    s = sq_name(mv[0]) + sq_name(mv[1])
    if mv[2]:
        s += PROMO_CH[mv[2]]
    return s


SAN_STRIP = str.maketrans("", "", "+#!?")
PIECE_OF_CH = {"N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN, "K": KING}
SAN_RE = re.compile(r"^([KQRBN])?([a-h])?([1-8])?(x)?([a-h][1-8])(=?[QRBN])?$")


def san_to_move(pos, san):
    s = san.translate(SAN_STRIP).replace("0", "O")
    if s in ("O-O", "O-O-O"):
        targets = {6, 62} if s == "O-O" else {2, 58}
        for mv in pos.legal_moves():
            if pos.is_castle(mv) and mv[1] in targets:
                return mv
        return None

    m = SAN_RE.match(s)
    if not m:
        return None
    piece = PIECE_OF_CH.get(m.group(1), PAWN)
    from_file = ord(m.group(2)[0]) - 97 if m.group(2) else -1
    from_rank = int(m.group(3)) - 1 if m.group(3) else -1
    dest = m.group(5)
    promo = {"Q": QUEEN, "R": ROOK, "B": BISHOP, "N": KNIGHT}.get(
        (m.group(6) or "").lstrip("="), 0)
    to = (int(dest[1]) - 1) * 8 + (ord(dest[0]) - 97)

    matches = [mv for mv in pos.legal_moves()
               if mv[1] == to and mv[2] == promo
               and abs(pos.board[mv[0]]) == piece
               and (from_file < 0 or mv[0] % 8 == from_file)
               and (from_rank < 0 or mv[0] // 8 == from_rank)]
    return matches[0] if matches else None


RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
TOKEN_RE = re.compile(r"\{[^}]*\}|;[^\n]*|\(|\)|\$\d+|[^\s(){};]+")


def split_games(text):
    games = []
    headers, movetext, in_moves = {}, [], False
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("[") and t.endswith("]") and not in_moves:
            key, _, val = t[1:].partition(" ")
            headers[key.strip()] = val.strip(' "')
        else:
            if t.startswith("[") and not in_moves:
                continue
            in_moves = True
            movetext.append(t)
            if any(r in t for r in RESULTS):
                games.append((headers, " ".join(movetext)))
                headers, movetext, in_moves = {}, [], False
    if in_moves or movetext:
        games.append((headers, " ".join(movetext)))
    return games


def tokenize(movetext):
    toks = []
    for m in TOKEN_RE.finditer(movetext):
        t = m.group(0)
        if t.startswith("{") or t.startswith(";") or t.startswith("$"):
            continue
        toks.append(t)
    return toks


def process_game(movetext, counts, max_ply, mainline_only):
    tokens = tokenize(movetext)
    state = {"i": 0}

    def record(pos, mv):
        h = pos.hash()
        u = mv_uci(mv)
        counts.setdefault(h, {}).setdefault(u, 0)
        counts[h][u] += 1

    def walk(pos, ply):
        before = None
        while state["i"] < len(tokens):
            t = tokens[state["i"]]
            if t == ")":
                return
            if t == "(":
                state["i"] += 1
                if mainline_only:
                    depth = 1
                    while state["i"] < len(tokens) and depth > 0:
                        if tokens[state["i"]] == "(":
                            depth += 1
                        elif tokens[state["i"]] == ")":
                            depth -= 1
                        state["i"] += 1
                    continue
                if before is not None:
                    walk(before.copy(), ply - 1)
                if state["i"] < len(tokens) and tokens[state["i"]] == ")":
                    state["i"] += 1
                continue
            if t in RESULTS or re.fullmatch(r"(\d+\.+|\.{3})", t):
                state["i"] += 1
                continue
            mv = san_to_move(pos, t)
            state["i"] += 1
            if mv is None:
                continue
            if ply < max_ply:
                record(pos, mv)
            before = pos.copy()
            pos.make(mv)
            ply += 1
        return

    walk(Position.from_fen(FEN_START), 0)
    return state["i"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PGN -> Waldemar-Opening-Book (WBF)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--max-ply", type=int, default=20,
                    help="Half-moves up to which recording takes place (default: 20)")
    ap.add_argument("--mainlines", action="store_true",
                    help="Ignore variations, main lines only")
    args = ap.parse_args()

    try:
        with open(args.input, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"Error: {e}")

    counts = {}
    games = split_games(text)
    plies = 0
    for headers, movetext in games:
        plies += process_game(movetext, counts, args.max_ply,
                              args.mainlines)

    lines = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for h, moves in counts.items():
            for uci, cnt in moves.items():
                f.write(f"{h:x} {uci} {cnt}\n")
                lines += 1

    print(f"{len(games)} Games read, {plies} Half-moves played")
    print(f"{lines} Book lines ({len(counts)} Positions) -> {args.output}")


if __name__ == "__main__":
    main()