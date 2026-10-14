package main

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	white = 1
	black = -1

	empty  = 0
	pawn   = 1
	knight = 2
	bishop = 3
	rook   = 4
	queen  = 5
	king   = 6

	whiteIdx = 0
	blackIdx = 1

	castleWK = 1 << 0
	castleWQ = 1 << 1
	castleBK = 1 << 2
	castleBQ = 1 << 3

	mateScore = 32000
	mateBound = mateScore - 512
	maxPly    = 128
	infinity  = 50000

	ttFlagExact = 0
	ttFlagLower = 1
	ttFlagUpper = 2

	maxHistory  = 1 << 15
	ttShardCnt  = 1024
	ttEntrySize = 64 // conservative estimate of sizeof(TTEntry) for sizing
)

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

type Move struct {
	From      int
	To        int
	Promo     int
	EnPassant bool
	Castle    bool
	Captured  int
}

func sameMove(a, b Move) bool {
	return a.From == b.From && a.To == b.To && a.Promo == b.Promo &&
		a.EnPassant == b.EnPassant && a.Castle == b.Castle
}

// Undo stores only cheap scalars; bitboards / eval are restored incrementally.
type Undo struct {
	captured int
	castling int
	ep       int
	halfmove int
	fullmove int
	hash     uint64
	kingSq   [2]int
}

type Position struct {
	board          [64]int
	sideToMove     int
	castlingRights int
	epSquare       int
	halfmove       int
	fullmove       int
	hash           uint64
	kingSq         [2]int

	byColor [2]uint64
	byType  [2][7]uint64

	// incremental tapered evaluation (white-positive)
	mg    int
	eg    int
	phase int
}

type SearchLimits struct {
	Depth    int
	MoveTime time.Duration
	Infinite bool
	WTime    time.Duration
	BTime    time.Duration
	WInc     time.Duration
	BInc     time.Duration
	Silent   bool
	Threads  int
}

type Engine struct {
	mu       sync.Mutex
	pos      Position
	cancel   context.CancelFunc
	busy     bool
	wg       sync.WaitGroup
	threads  int
	hashMB   int
	book     *Book
	ownBook  bool
	bookFile string
}

type MoveList struct {
	moves  [256]Move
	scores [256]int
	count  int
}

func (ml *MoveList) reset()     { ml.count = 0 }
func (ml *MoveList) add(m Move) { ml.moves[ml.count] = m; ml.count++ }

// ---------------------------------------------------------------------------
// Evaluation tables (tapered, midgame / endgame)
// ---------------------------------------------------------------------------

var (
	phaseWeight = [7]int{0, 0, 1, 1, 2, 4, 0}

	mgValue = [7]int{0, 82, 300, 350, 477, 1025, 0}
	egValue = [7]int{0, 94, 280, 320, 512, 936, 0}

	// Midgame piece-square tables (white perspective, a1 = index 0).
	mgPst = [7][64]int{
		{},
		{
			0, 0, 0, 0, 0, 0, 0, 0,
			98, 134, 61, 95, 68, 126, 34, -11,
			-6, 7, 26, 31, 65, 56, 25, -20,
			-14, 13, 6, 21, 23, 12, 17, -23,
			-27, -2, -5, 12, 17, 6, 10, -25,
			-26, -4, -4, -10, 3, 3, 33, -12,
			-35, -1, -20, -23, -15, 24, 38, -22,
			0, 0, 0, 0, 0, 0, 0, 0,
		},
		{
			-167, -89, -34, -49, 61, -97, -15, -107,
			-73, -41, 72, 36, 23, 62, 7, -17,
			-47, 60, 37, 65, 84, 129, 73, 32,
			-9, 17, 19, 53, 37, 69, 18, 22,
			-13, 4, 16, 13, 28, 19, 21, -8,
			-23, -9, 12, 10, 19, 17, 25, -16,
			-29, -53, -12, -3, -1, 18, -14, -19,
			-105, -21, -58, -33, -17, -28, -19, -23,
		},
		{
			-29, 4, -82, -37, -25, -42, 7, -8,
			-26, 16, -18, -13, 30, 59, 18, -47,
			-16, 37, 43, 40, 35, 50, 37, -2,
			-4, 5, 19, 50, 37, 37, 15, -4,
			-6, 13, 13, 26, 34, 12, 10, -5,
			0, 15, 15, 15, 14, 27, 18, 10,
			4, 15, 16, 0, 7, 21, 33, 1,
			-33, -3, -14, -21, -13, -12, -39, -21,
		},
		{
			32, 42, 32, 51, 63, 9, 31, 43,
			27, 32, 58, 62, 80, 67, 26, 44,
			-5, 19, 26, 36, 17, 45, 61, 16,
			-24, -11, 7, 26, 24, 35, -8, -20,
			-36, -26, -12, -1, 9, -7, -6, -23,
			-45, -25, -16, -17, 3, 0, -5, -33,
			-44, -16, -20, -9, -1, 16, 0, -37,
			-19, -13, 1, 17, 16, 7, -37, -26,
		},
		{
			-28, 0, 29, 12, 59, 44, 43, 45,
			-24, -39, -5, 1, -16, 57, 28, 54,
			-13, -17, 7, 8, 29, 56, 47, 57,
			-27, -27, -16, -16, -1, 17, -2, 1,
			-9, -26, -9, -10, -2, -4, 3, -3,
			-14, 2, -11, -2, -5, 2, 14, 5,
			-35, -8, 11, 2, 8, 15, -3, 1,
			-1, -18, -9, 10, -15, -25, -31, -50,
		},
		{
			-65, 23, 16, -15, -56, -34, 2, 13,
			29, -1, -20, -7, -8, -4, -38, -29,
			-9, 24, 2, -16, -20, 6, 22, -22,
			-17, -20, -12, -27, -30, -25, -14, -36,
			-49, -1, -27, -39, -46, -44, -33, -51,
			-14, -14, -22, -46, -44, -30, -15, -27,
			1, 7, -8, -64, -43, -16, 9, 8,
			-15, 36, 12, -54, 8, -28, 24, 14,
		},
	}

	// Endgame piece-square tables (white perspective, a1 = index 0).
	egPst = [7][64]int{
		{},
		{
			0, 0, 0, 0, 0, 0, 0, 0,
			178, 173, 158, 134, 147, 132, 165, 187,
			94, 100, 85, 67, 56, 53, 82, 84,
			32, 24, 13, 5, -2, 4, 17, 17,
			13, 9, -3, -7, -7, -8, 3, -1,
			4, 7, -6, 1, 0, -5, -1, -8,
			13, 8, 8, 10, 13, 0, 2, -7,
			0, 0, 0, 0, 0, 0, 0, 0,
		},
		{
			-58, -38, -13, -28, -31, -27, -63, -99,
			-25, -8, -25, -2, -9, -25, -24, -52,
			-24, -20, 10, 9, -1, -9, -19, -41,
			-17, 3, 22, 22, 22, 11, 8, -18,
			-18, -6, 16, 25, 16, 17, 4, -18,
			-23, -3, -1, 15, 10, -3, -20, -22,
			-42, -20, -10, -5, -2, -20, -23, -44,
			-29, -51, -23, -15, -22, -18, -50, -64,
		},
		{
			-14, -21, -11, -8, -7, -9, -17, -24,
			-8, -4, 7, -12, -3, -13, -4, -14,
			2, -8, 0, -20, -14, 0, -13, -10,
			-3, 10, 10, -14, -5, 3, 7, -5,
			-6, 5, 3, -4, -6, 1, 0, 4,
			-10, 4, 5, -11, 1, -2, 3, -6,
			-4, 0, -10, -6, -4, -6, 6, -8,
			-17, -15, -15, -19, -20, -15, -23, -23,
		},
		{
			13, 10, 18, 15, 12, 12, 8, 5,
			11, 13, 13, 11, -3, 3, 8, 3,
			7, 7, 7, 5, 4, -3, -5, -3,
			4, 3, 13, 1, 2, 1, -1, 2,
			3, 5, 8, 4, -5, -6, -8, -11,
			-4, 0, -5, -1, -7, -12, -4, -11,
			-6, -6, 0, 2, -9, -9, -11, -3,
			-9, 2, 3, -1, -5, -13, -6, -12,
		},
		{
			-9, 22, 22, 27, 27, 19, 22, 23,
			-20, 6, 6, 4, 4, 6, 8, 5,
			-3, -1, 5, 10, 14, 6, 5, 1,
			-5, -11, -5, -3, 2, -9, -3, -12,
			-9, -5, -7, 3, -3, -6, -7, -12,
			-13, -2, -3, -5, -4, -2, -7, -13,
			-10, -6, -4, -3, -3, -5, -9, -13,
			-7, -9, -6, -4, -5, -9, -11, -15,
		},
		{
			-74, -35, -18, -18, -11, 15, 4, -17,
			-12, 17, 14, 17, 17, 38, 23, 11,
			10, 17, 23, 15, 20, 45, 44, 13,
			-8, 22, 24, 27, 26, 33, 26, 3,
			-18, -4, 21, 24, 27, 23, 9, -11,
			-19, -3, 11, 21, 23, 16, 7, -9,
			-27, -11, 4, 13, 14, 4, -5, -17,
			-53, -34, -21, -11, -13, -18, -32, -43,
		},
	}

	passedBonus = [8]int{0, 8, 16, 28, 45, 75, 120, 0}

	knightOffsets = [8]int{17, 15, 10, 6, -17, -15, -10, -6}
	bishopDirs    = [4]int{9, 7, -9, -7}
	rookDirs      = [4]int{8, -8, 1, -1}
	queenDirs     = [8]int{9, 7, -9, -7, 8, -8, 1, -1}
	kingOffsets   = [8]int{8, -8, 1, -1, 9, 7, -9, -7}
	promoPieces   = [4]int{queen, rook, bishop, knight}

	knightMask    [64]uint64
	kingMask      [64]uint64
	pawnMask      [2][64]uint64
	pawnAttackers [2][64]uint64
	fileMask      [8]uint64
	adjFileMask   [8]uint64
	passedMask    [2][64]uint64

	lmrTable [64][64]int

	zobristPiece    [12][64]uint64
	zobristCastling [16]uint64
	zobristEPFile   [9]uint64
	zobristSide     uint64
)

func init() {
	initAttackTables()
	initZobrist()
	for d := 1; d < 64; d++ {
		for m := 1; m < 64; m++ {
			v := 0.75 + math.Log(float64(d))*math.Log(float64(m))/2.25
			if v < 0 {
				v = 0
			}
			lmrTable[d][m] = int(v)
		}
	}
	for f := 0; f < 8; f++ {
		fileMask[f] = uint64(0x0101010101010101) << f
		if f > 0 {
			adjFileMask[f] |= fileMask[f-1]
		}
		if f < 7 {
			adjFileMask[f] |= fileMask[f+1]
		}
	}
	for sq := 0; sq < 64; sq++ {
		f := sq % 8
		r := sq / 8
		// White: all squares in files f-1..f+1 with rank > r.
		for rr := r + 1; rr < 8; rr++ {
			for ff := f - 1; ff <= f+1; ff++ {
				if ff >= 0 && ff < 8 {
					passedMask[whiteIdx][sq] |= uint64(1) << (rr*8 + ff)
				}
			}
		}
		for rr := r - 1; rr >= 0; rr-- {
			for ff := f - 1; ff <= f+1; ff++ {
				if ff >= 0 && ff < 8 {
					passedMask[blackIdx][sq] |= uint64(1) << (rr*8 + ff)
				}
			}
		}
	}
}

func initAttackTables() {
	for sq := 0; sq < 64; sq++ {
		for _, off := range knightOffsets {
			to := sq + off
			if inBoard(to) && knightPattern(sq, to) {
				knightMask[sq] |= uint64(1) << to
			}
		}
		for _, off := range kingOffsets {
			to := sq + off
			if inBoard(to) && kingPattern(sq, to) {
				kingMask[sq] |= uint64(1) << to
			}
		}
		f := sq % 8
		if f > 0 && inBoard(sq+7) {
			pawnMask[whiteIdx][sq] |= uint64(1) << (sq + 7)
		}
		if f < 7 && inBoard(sq+9) {
			pawnMask[whiteIdx][sq] |= uint64(1) << (sq + 9)
		}
		if f > 0 && inBoard(sq-9) {
			pawnMask[blackIdx][sq] |= uint64(1) << (sq - 9)
		}
		if f < 7 && inBoard(sq-7) {
			pawnMask[blackIdx][sq] |= uint64(1) << (sq - 7)
		}
		// Reverse: squares from which a pawn of that color can attack sq
		if f > 0 && inBoard(sq-9) {
			pawnAttackers[whiteIdx][sq] |= uint64(1) << (sq - 9)
		}
		if f < 7 && inBoard(sq-7) {
			pawnAttackers[whiteIdx][sq] |= uint64(1) << (sq - 7)
		}
		if f > 0 && inBoard(sq+7) {
			pawnAttackers[blackIdx][sq] |= uint64(1) << (sq + 7)
		}
		if f < 7 && inBoard(sq+9) {
			pawnAttackers[blackIdx][sq] |= uint64(1) << (sq + 9)
		}
	}
}

func initZobrist() {
	var x uint64 = 0x9e3779b97f4a7c15
	next := func() uint64 {
		x ^= x >> 12
		x ^= x << 25
		x ^= x >> 27
		return x * 2685821657736338717
	}
	for i := range zobristPiece {
		for j := 0; j < 64; j++ {
			zobristPiece[i][j] = next()
		}
	}
	for i := 0; i < 16; i++ {
		zobristCastling[i] = next()
	}
	for i := 0; i < 9; i++ {
		zobristEPFile[i] = next()
	}
	zobristSide = next()
}

// ---------------------------------------------------------------------------
// Transposition table (sharded, depth-preferred with generation aging)
// ---------------------------------------------------------------------------

type TTEntry struct {
	key  uint64
	move Move
	//
	//g score is stored relative to the node (ply 0), mate scores already adjusted.
	score int
	depth int8
	flag  uint8
	gen   uint8
}

type ttShard struct {
	mu      sync.RWMutex
	entries []TTEntry
}

type sharedTT struct {
	shards [ttShardCnt]ttShard
	gen    uint8
}

func newSharedTT(mb int) *sharedTT {
	if mb < 1 {
		mb = 1
	}
	total := mb << 20 / ttEntrySize
	if total < ttShardCnt {
		total = ttShardCnt
	}
	perShard := 1
	for perShard*ttShardCnt < total {
		perShard <<= 1
	}
	tt := &sharedTT{}
	for i := range tt.shards {
		tt.shards[i].entries = make([]TTEntry, perShard)
	}
	return tt
}

func (t *sharedTT) shard(hash uint64) *ttShard {
	return &t.shards[uint64(hash>>32)%ttShardCnt]
}

func (t *sharedTT) nextGen() uint8 {
	for i := range &t.shards {
		t.shards[i].mu.Lock()
	}
	t.gen++
	if t.gen == 0 {
		t.gen = 1
	}
	g := t.gen
	for i := range &t.shards {
		t.shards[i].mu.Unlock()
	}
	return g
}

func (s *Searcher) probeTT(hash uint64, depth, alpha, beta, ply int) (int, bool, Move) {
	sh := s.tt.shard(hash)
	sh.mu.RLock()
	entry := sh.entries[int(hash)&(len(sh.entries)-1)]
	sh.mu.RUnlock()
	if entry.key != hash {
		return 0, false, Move{}
	}
	score := entry.score
	if score > mateBound {
		score -= ply
	} else if score < -mateBound {
		score += ply
	}
	if int(entry.depth) >= depth {
		switch entry.flag {
		case ttFlagExact:
			return score, true, entry.move
		case ttFlagLower:
			if score >= beta {
				return score, true, entry.move
			}
		case ttFlagUpper:
			if score <= alpha {
				return score, true, entry.move
			}
		}
	}
	return 0, false, entry.move
}

func (s *Searcher) storeTT(hash uint64, depth, score int, move Move, flag uint8, ply int) {
	if score > mateBound {
		score += ply
	} else if score < -mateBound {
		score -= ply
	}
	sh := s.tt.shard(hash)
	idx := int(hash) & (len(sh.entries) - 1)
	sh.mu.Lock()
	entry := &sh.entries[idx]
	if entry.key != hash || entry.gen != s.gen || int(entry.depth) <= depth || flag == ttFlagExact {
		if entry.key == hash && move.From < 0 && entry.move.From >= 0 {
			move = entry.move // keep old move if new one is empty
		}
		*entry = TTEntry{key: hash, move: move, score: score, depth: int8(depth), flag: flag, gen: s.gen}
	}
	sh.mu.Unlock()
}

// ---------------------------------------------------------------------------
// Searcher
// ---------------------------------------------------------------------------

type Searcher struct {
	tt         *sharedTT
	gen        uint8
	nodes      uint64
	history    [2][64][64]int
	killers    [maxPly][2]Move
	counter    [2][64][64]Move
	lastMoveTo [maxPly]int
	repeat     [1024]uint64
	repCount   int
	ctx        context.Context
	hardDead   time.Time
	softDead   time.Time
	stopped    bool
	start      time.Time
	lastScore  int
	lastDepth  int
	pvLine     []Move
}

// ---------------------------------------------------------------------------
// UCI main loop
// ---------------------------------------------------------------------------

func main() {
	threads := runtime.NumCPU()
	if threads < 1 {
		threads = 1
	}
	if threads > 128 {
		threads = 128
	}
	engine := &Engine{
		pos:      NewStartPosition(),
		threads:  threads,
		hashMB:   32,
		ownBook:  true,
		bookFile: "book.wbf",
	}
	if bk, err := loadBook(engine.bookFile); err == nil {
		engine.book = bk
	} else {
		fmt.Printf("info string failed to load book %s: %v\n", engine.bookFile, err)
	}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)

	if scanner != nil {
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			if engine.handleCommand(line) {
				break
			}
		}
	}
	engine.stopSearch()
	engine.wg.Wait()
}

func (e *Engine) handleCommand(line string) bool {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return false
	}
	switch fields[0] {
	case "uci":
		fmt.Println("id name Waldemar 2.1 Optimized")
		fmt.Println("id author zlElo")
		fmt.Printf("option name Threads type spin default %d min 1 max 128\n", e.threads)
		fmt.Println("option name Hash type spin default 32 min 1 max 4096")
		fmt.Println("option name OwnBook type check default true")
		fmt.Println("option name BookFile type string default book.wbf")
		fmt.Println("uciok")
	case "isready":
		fmt.Println("readyok")
	case "ucinewgame":
		e.stopSearch()
		e.wg.Wait()
		e.mu.Lock()
		e.pos = NewStartPosition()
		e.mu.Unlock()
	case "position":
		e.stopSearch()
		e.wg.Wait()
		e.mu.Lock()
		err := e.applyPosition(fields[1:])
		e.mu.Unlock()
		if err != nil {
			fmt.Fprintf(os.Stderr, "position error: %v\n", err)
		}
	case "setoption":
		e.setOption(fields[1:])
	case "go":
		limits := parseGo(fields[1:])
		e.startSearch(limits)
	case "stop":
		e.stopSearch()
	case "quit":
		e.stopSearch()
		return true
	}
	return false
}

func (e *Engine) applyPosition(fields []string) error {
	if len(fields) == 0 {
		return fmt.Errorf("missing position args")
	}
	idx := 0
	switch fields[0] {
	case "startpos":
		e.pos = NewStartPosition()
		idx = 1
	case "fen":
		if len(fields) < 6 {
			return fmt.Errorf("invalid fen")
		}
		fenFields := fields[1:]
		end := len(fenFields)
		if end > 6 {
			end = 6
		}
		pos, err := NewPositionFromFEN(strings.Join(fenFields[:end], " "))
		if err != nil {
			return err
		}
		e.pos = pos
		idx = 1 + end
	default:
		return fmt.Errorf("unknown position mode")
	}
	if idx < len(fields) && fields[idx] == "moves" {
		for _, moveStr := range fields[idx+1:] {
			move, ok := parseUCIMove(moveStr, &e.pos)
			if !ok {
				return fmt.Errorf("illegal move %s", moveStr)
			}
			e.pos.MakeMove(move)
		}
	}
	return nil
}

func parseGo(fields []string) SearchLimits {
	limits := SearchLimits{Depth: maxPly - 1}
	for i := 0; i < len(fields); i++ {
		next := func() int {
			if i+1 < len(fields) {
				if v, err := strconv.Atoi(fields[i+1]); err == nil {
					return v
				}
			}
			return 0
		}
		switch fields[i] {
		case "depth":
			if v := next(); v > 0 {
				limits.Depth = v
			}
		case "movetime":
			if v := next(); v > 0 {
				limits.MoveTime = time.Duration(v) * time.Millisecond
			}
		case "infinite":
			limits.Infinite = true
			limits.Depth = maxPly - 1
		case "wtime":
			if v := next(); v > 0 {
				limits.WTime = time.Duration(v) * time.Millisecond
			}
		case "btime":
			if v := next(); v > 0 {
				limits.BTime = time.Duration(v) * time.Millisecond
			}
		case "winc":
			if v := next(); v >= 0 {
				limits.WInc = time.Duration(v) * time.Millisecond
			}
		case "binc":
			if v := next(); v >= 0 {
				limits.BInc = time.Duration(v) * time.Millisecond
			}
		case "nodes":
			// parsed but ignored
		}
	}
	// The go parser consumed one extra token for value keywords; adjust loop.
	// (Handled inline by the switch; no-op.)
	return limits
}

func (e *Engine) setOption(fields []string) {
	if len(fields) < 4 || !strings.EqualFold(fields[0], "name") {
		return
	}
	name := fields[1]
	value := ""
	for i, f := range fields {
		if strings.EqualFold(f, "value") && i+1 < len(fields) {
			value = fields[i+1]
		}
	}
	if value == "" {
		return
	}
	nameLower := strings.ToLower(name)
	switch nameLower {
	case "threads":
		v, err := strconv.Atoi(strings.TrimSpace(value))
		if err != nil {
			return
		}
		if v < 1 {
			v = 1
		}
		if v > 128 {
			v = 128
		}
		e.stopSearch()
		e.wg.Wait()
		e.mu.Lock()
		e.threads = v
		e.mu.Unlock()
	case "hash":
		v, err := strconv.Atoi(strings.TrimSpace(value))
		if err != nil {
			return
		}
		if v < 1 {
			v = 1
		}
		if v > 4096 {
			v = 4096
		}
		e.stopSearch()
		e.wg.Wait()
		e.mu.Lock()
		e.hashMB = v
		e.mu.Unlock()
	case "ownbook":
		e.stopSearch()
		e.wg.Wait()
		e.mu.Lock()
		e.ownBook = strings.EqualFold(value, "true")
		e.mu.Unlock()
	case "bookfile":
		e.stopSearch()
		e.wg.Wait()
		e.mu.Lock()
		e.bookFile = value
		if bk, err := loadBook(e.bookFile); err == nil {
			e.book = bk
		} else {
			e.book = nil
			fmt.Printf("info string failed to load book %s: %v\n", e.bookFile, err)
		}
		e.mu.Unlock()
	}
}

func (e *Engine) startSearch(l SearchLimits) {
	e.stopSearch()
	e.wg.Wait()

	e.mu.Lock()
	pos := e.pos
	threads := e.threads
	hashMB := e.hashMB
	useBook := e.ownBook && e.book != nil
	book := e.book
	e.mu.Unlock()

	// Opening book probe
	if useBook {
		fmt.Printf("info string book probe hash=%016x\n", pos.hash)
		if m, ok := book.getMove(pos.hash, &pos); ok {
			fmt.Println("info string book hit")
			fmt.Printf("bestmove %s\n", m.UCI())
			return
		}
	} else {
		fmt.Println("info string book disabled or not loaded")
	}

	if l.Threads < 1 {
		l.Threads = threads
	}

	ctx, cancel := context.WithCancel(context.Background())
	e.mu.Lock()
	e.cancel = cancel
	e.busy = true
	e.mu.Unlock()

	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		best := SearchBestMove(ctx, pos, l, hashMB)
		if best.From >= 0 {
			fmt.Printf("bestmove %s\n", best.UCI())
		} else {
			fmt.Println("bestmove 0000")
		}
		e.mu.Lock()
		e.busy = false
		e.cancel = nil
		e.mu.Unlock()
	}()
}

func (e *Engine) stopSearch() {
	e.mu.Lock()
	cancel := e.cancel
	e.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// ---------------------------------------------------------------------------
// FEN / position setup
// ---------------------------------------------------------------------------

func NewStartPosition() Position {
	pos, _ := NewPositionFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	return pos
}

func NewPositionFromFEN(fen string) (Position, error) {
	var p Position
	p.epSquare = -1
	p.kingSq[0] = -1
	p.kingSq[1] = -1
	parts := strings.Fields(fen)
	if len(parts) < 2 {
		return p, fmt.Errorf("invalid fen")
	}

	ranks := strings.Split(parts[0], "/")
	if len(ranks) != 8 {
		return p, fmt.Errorf("invalid fen board")
	}
	for r := 0; r < 8; r++ {
		file := 0
		for _, ch := range ranks[r] {
			if ch >= '1' && ch <= '8' {
				file += int(ch - '0')
				continue
			}
			if file >= 8 {
				return p, fmt.Errorf("invalid fen rank")
			}
			sq := (7-r)*8 + file
			piece := fenRuneToPiece(ch)
			if piece == empty {
				return p, fmt.Errorf("invalid fen piece")
			}
			p.board[sq] = piece
			p.addPiece(piece, sq)
			switch piece {
			case king:
				p.kingSq[whiteIdx] = sq
			case -king:
				p.kingSq[blackIdx] = sq
			}
			file++
		}
		if file != 8 {
			return p, fmt.Errorf("invalid fen file count")
		}
	}

	switch parts[1] {
	case "w":
		p.sideToMove = white
	case "b":
		p.sideToMove = black
	default:
		return p, fmt.Errorf("invalid side to move")
	}

	if len(parts) > 2 && parts[2] != "-" {
		if strings.ContainsRune(parts[2], 'K') {
			p.castlingRights |= castleWK
		}
		if strings.ContainsRune(parts[2], 'Q') {
			p.castlingRights |= castleWQ
		}
		if strings.ContainsRune(parts[2], 'k') {
			p.castlingRights |= castleBK
		}
		if strings.ContainsRune(parts[2], 'q') {
			p.castlingRights |= castleBQ
		}
	}
	if len(parts) > 3 && parts[3] != "-" {
		sq, ok := squareFromString(parts[3])
		if !ok {
			return p, fmt.Errorf("invalid ep square")
		}
		p.epSquare = sq
	}
	if len(parts) > 4 {
		if v, err := strconv.Atoi(parts[4]); err == nil {
			p.halfmove = v
		}
	}
	if len(parts) > 5 {
		if v, err := strconv.Atoi(parts[5]); err == nil {
			p.fullmove = v
		}
	}
	if p.fullmove == 0 {
		p.fullmove = 1
	}
	p.hash = p.computeHash()
	return p, nil
}

func fenRuneToPiece(ch rune) int {
	switch ch {
	case 'P':
		return pawn
	case 'N':
		return knight
	case 'B':
		return bishop
	case 'R':
		return rook
	case 'Q':
		return queen
	case 'K':
		return king
	case 'p':
		return -pawn
	case 'n':
		return -knight
	case 'b':
		return -bishop
	case 'r':
		return -rook
	case 'q':
		return -queen
	case 'k':
		return -king
	default:
		return empty
	}
}

// ---------------------------------------------------------------------------
// Piece state helpers (incremental bitboards + tapered eval)
// ---------------------------------------------------------------------------

func (p *Position) addPiece(piece, sq int) {
	ci := blackIdx
	if piece > 0 {
		ci = whiteIdx
	}
	pt := abs(piece)
	bb := uint64(1) << sq
	p.byColor[ci] |= bb
	p.byType[ci][pt] |= bb
	idx := sq
	if ci == blackIdx {
		idx = mirrorSq(sq)
	}
	p.mg += colorOf(piece) * (mgValue[pt] + mgPst[pt][idx])
	p.eg += colorOf(piece) * (egValue[pt] + egPst[pt][idx])
	p.phase += phaseWeight[pt]
}

func (p *Position) removePiece(piece, sq int) {
	ci := blackIdx
	if piece > 0 {
		ci = whiteIdx
	}
	pt := abs(piece)
	bb := uint64(1) << sq
	p.byColor[ci] &^= bb
	p.byType[ci][pt] &^= bb
	idx := sq
	if ci == blackIdx {
		idx = mirrorSq(sq)
	}
	p.mg -= colorOf(piece) * (mgValue[pt] + mgPst[pt][idx])
	p.eg -= colorOf(piece) * (egValue[pt] + egPst[pt][idx])
	p.phase -= phaseWeight[pt]
}

func (p *Position) movePiece(piece, from, to int) {
	p.removePiece(piece, from)
	p.addPiece(piece, to)
}

// ---------------------------------------------------------------------------
// Make / Undo (incremental, no bitboard copies)
// ---------------------------------------------------------------------------

func (p *Position) MakeMove(m Move) Undo {
	u := Undo{
		captured: p.board[m.To],
		castling: p.castlingRights,
		ep:       p.epSquare,
		halfmove: p.halfmove,
		fullmove: p.fullmove,
		hash:     p.hash,
		kingSq:   p.kingSq,
	}
	piece := p.board[m.From]
	movingType := abs(piece)
	moverIdx := blackIdx
	if p.sideToMove == white {
		moverIdx = whiteIdx
	}

	p.epSquare = -1
	if movingType == pawn || u.captured != empty || m.EnPassant {
		p.halfmove = 0
	} else {
		p.halfmove++
	}
	if p.sideToMove == black {
		p.fullmove++
	}

	p.hash ^= zobristCastling[u.castling] ^ zobristEPFile[epFileIndex(u.ep)] ^
		zobristPiece[pieceHashIdx(piece)][m.From] ^ zobristSide
	p.removePiece(piece, m.From)
	p.board[m.From] = empty

	if m.EnPassant {
		capSq := m.To - 8
		if p.sideToMove == black {
			capSq = m.To + 8
		}
		u.captured = p.board[capSq]
		p.hash ^= zobristPiece[pieceHashIdx(u.captured)][capSq]
		p.removePiece(u.captured, capSq)
		p.board[capSq] = empty
	} else if u.captured != empty {
		p.hash ^= zobristPiece[pieceHashIdx(u.captured)][m.To]
		p.removePiece(u.captured, m.To)
	}

	if m.Castle {
		switch m.To {
		case 6: // white O-O: rook h1 -> f1
			r := p.board[7]
			p.hash ^= zobristPiece[pieceHashIdx(r)][7] ^ zobristPiece[pieceHashIdx(r)][5]
			p.board[5], p.board[7] = r, empty
			p.movePiece(r, 7, 5)
		case 2: // white O-O-O: rook a1 -> d1
			r := p.board[0]
			p.hash ^= zobristPiece[pieceHashIdx(r)][0] ^ zobristPiece[pieceHashIdx(r)][3]
			p.board[3], p.board[0] = r, empty
			p.movePiece(r, 0, 3)
		case 62: // black O-O
			r := p.board[63]
			p.hash ^= zobristPiece[pieceHashIdx(r)][63] ^ zobristPiece[pieceHashIdx(r)][61]
			p.board[61], p.board[63] = r, empty
			p.movePiece(r, 63, 61)
		case 58: // black O-O-O
			r := p.board[56]
			p.hash ^= zobristPiece[pieceHashIdx(r)][56] ^ zobristPiece[pieceHashIdx(r)][59]
			p.board[59], p.board[56] = r, empty
			p.movePiece(r, 56, 59)
		}
	}

	if m.Promo != empty {
		if p.sideToMove == white {
			piece = m.Promo
		} else {
			piece = -m.Promo
		}
	}
	p.board[m.To] = piece
	p.addPiece(piece, m.To)
	p.hash ^= zobristPiece[pieceHashIdx(piece)][m.To]

	if movingType == pawn && abs(m.To-m.From) == 16 {
		if p.sideToMove == white {
			p.epSquare = m.From + 8
		} else {
			p.epSquare = m.From - 8
		}
	}

	if movingType == king {
		p.kingSq[moverIdx] = m.To
	}

	p.updateCastling(m, movingType, u.captured)
	p.hash ^= zobristCastling[p.castlingRights] ^ zobristEPFile[epFileIndex(p.epSquare)]
	p.sideToMove = -p.sideToMove
	return u
}

func (p *Position) UndoMove(m Move, u Undo) {
	p.sideToMove = -p.sideToMove
	p.castlingRights = u.castling
	p.epSquare = u.ep
	p.halfmove = u.halfmove
	p.fullmove = u.fullmove
	p.hash = u.hash
	p.kingSq = u.kingSq

	// Remove the piece currently standing on m.To from bitboards/eval.
	p.removePiece(p.board[m.To], m.To)

	if m.Castle {
		var rook int
		var rookFrom, rookTo int
		switch m.To {
		case 6:
			rook, rookFrom, rookTo = p.board[5], 5, 7
		case 2:
			rook, rookFrom, rookTo = p.board[3], 3, 0
		case 62:
			rook, rookFrom, rookTo = p.board[61], 61, 63
		case 58:
			rook, rookFrom, rookTo = p.board[59], 59, 56
		}
		p.removePiece(rook, rookFrom)
		p.board[rookTo] = rook
		p.board[rookFrom] = empty
		p.addPiece(rook, rookTo)
	}

	moved := p.board[m.To]
	if m.Promo != empty {
		if p.sideToMove == white {
			moved = pawn
		} else {
			moved = -pawn
		}
	}
	p.board[m.From] = moved
	p.addPiece(moved, m.From)

	if m.EnPassant {
		p.board[m.To] = empty
		capSq := m.To - 8
		if p.sideToMove == black {
			capSq = m.To + 8
		}
		p.board[capSq] = u.captured
		p.addPiece(u.captured, capSq)
	} else if u.captured != empty {
		p.board[m.To] = u.captured
		p.addPiece(u.captured, m.To)
	} else {
		p.board[m.To] = empty
	}
}

func (p *Position) MakeNullMove() Undo {
	u := Undo{castling: p.castlingRights, ep: p.epSquare, halfmove: p.halfmove,
		fullmove: p.fullmove, hash: p.hash, kingSq: p.kingSq}
	p.hash ^= zobristEPFile[epFileIndex(p.epSquare)] ^ zobristEPFile[0] ^ zobristSide
	p.epSquare = -1
	if p.sideToMove == black {
		p.fullmove++
	}
	p.halfmove++
	p.sideToMove = -p.sideToMove
	return u
}

func (p *Position) UndoNullMove(u Undo) {
	p.sideToMove = -p.sideToMove
	p.castlingRights = u.castling
	p.epSquare = u.ep
	p.halfmove = u.halfmove
	p.fullmove = u.fullmove
	p.hash = u.hash
	p.kingSq = u.kingSq
}

func (p *Position) updateCastling(m Move, movingType, captured int) {
	switch m.From {
	case 4:
		p.castlingRights &^= castleWK | castleWQ
	case 60:
		p.castlingRights &^= castleBK | castleBQ
	case 0:
		p.castlingRights &^= castleWQ
	case 7:
		p.castlingRights &^= castleWK
	case 56:
		p.castlingRights &^= castleBQ
	case 63:
		p.castlingRights &^= castleBK
	}
	if captured != empty {
		switch m.To {
		case 0:
			p.castlingRights &^= castleWQ
		case 7:
			p.castlingRights &^= castleWK
		case 56:
			p.castlingRights &^= castleBQ
		case 63:
			p.castlingRights &^= castleBK
		}
	}
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

func pieceHashIdx(piece int) int {
	if piece > 0 {
		return piece - 1
	}
	return 6 + (-piece - 1)
}

func epFileIndex(ep int) int {
	if ep < 0 || ep >= 64 {
		return 0
	}
	return (ep % 8) + 1
}

func parseUCIMove(moveStr string, pos *Position) (Move, bool) {
	if len(moveStr) < 4 {
		return Move{}, false
	}
	from, ok1 := squareFromString(moveStr[:2])
	to, ok2 := squareFromString(moveStr[2:4])
	if !ok1 || !ok2 {
		return Move{}, false
	}
	promo := empty
	if len(moveStr) == 5 {
		switch moveStr[4] {
		case 'q':
			promo = queen
		case 'r':
			promo = rook
		case 'b':
			promo = bishop
		case 'n':
			promo = knight
		default:
			return Move{}, false
		}
	}
	var legal MoveList
	pos.generateLegalTo(&legal, false)
	for i := 0; i < legal.count; i++ {
		m := legal.moves[i]
		if m.From == from && m.To == to && m.Promo == promo {
			return m, true
		}
	}
	return Move{}, false
}

func (m Move) UCI() string {
	if m.From < 0 || m.To < 0 {
		return "0000"
	}
	s := squareToString(m.From) + squareToString(m.To)
	if m.Promo != empty {
		s += string(promoRune(m.Promo))
	}
	return s
}

func promoRune(piece int) rune {
	switch piece {
	case queen:
		return 'q'
	case rook:
		return 'r'
	case bishop:
		return 'b'
	case knight:
		return 'n'
	}
	return 'q'
}

func squareFromString(s string) (int, bool) {
	if len(s) != 2 {
		return 0, false
	}
	file := int(s[0] - 'a')
	rank := int(s[1] - '1')
	if file < 0 || file > 7 || rank < 0 || rank > 7 {
		return 0, false
	}
	return rank*8 + file, true
}

func squareToString(sq int) string {
	return string([]byte{byte('a' + sq%8), byte('1' + sq/8)})
}

func mirrorSq(sq int) int { return sq ^ 56 }
func inBoard(sq int) bool { return sq >= 0 && sq < 64 }
func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
func colorOf(piece int) int {
	if piece > 0 {
		return white
	}
	return black
}

func knightPattern(from, to int) bool {
	df := abs((from % 8) - (to % 8))
	dr := abs((from / 8) - (to / 8))
	return (df == 1 && dr == 2) || (df == 2 && dr == 1)
}

func kingPattern(from, to int) bool {
	df := abs((from % 8) - (to % 8))
	dr := abs((from / 8) - (to / 8))
	return df <= 1 && dr <= 1
}

func sameLine(from, to, dir int) bool {
	ff := from % 8
	tf := to % 8
	switch dir {
	case 1, -1:
		return abs(tf-ff) == 1
	case 8, -8:
		return true
	case 9, -9, 7, -7:
		return abs(tf-ff) == 1
	}
	return false
}
