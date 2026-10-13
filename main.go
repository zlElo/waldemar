package main

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"path/filepath"
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
// Opening book (.wbf format: hexhash ucimove weight)
// ---------------------------------------------------------------------------

type BookMove struct {
	uci    string
	weight int
}

type Book struct {
	moves map[uint64][]BookMove
}

func loadBook(filename string) (*Book, error) {
	// Try given path, then next to executable
	paths := []string{filename}
	if exe, err := os.Executable(); err == nil {
		paths = append(paths, filepath.Join(filepath.Dir(exe), filename))
	}
	var f *os.File
	var err error
	var used string
	for _, p := range paths {
		f, err = os.Open(p)
		if err == nil {
			used = p
			break
		}
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	b := &Book{moves: make(map[uint64][]BookMove)}
	scanner := bufio.NewScanner(f)
	count := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}
		hash, err := strconv.ParseUint(parts[0], 16, 64)
		if err != nil {
			continue
		}
		weight, err := strconv.Atoi(parts[2])
		if err != nil || weight <= 0 {
			continue
		}
		b.moves[hash] = append(b.moves[hash], BookMove{uci: parts[1], weight: weight})
		count++
	}
	fmt.Printf("info string book loaded from %s (%d entries, %d positions)\n", used, count, len(b.moves))
	return b, scanner.Err()
}

func (b *Book) getMove(hash uint64, pos *Position) (Move, bool) {
	if b == nil {
		return Move{}, false
	}
	entries, ok := b.moves[hash]
	if !ok || len(entries) == 0 {
		return Move{}, false
	}
	total := 0
	for _, e := range entries {
		total += e.weight
	}
	if total <= 0 {
		return Move{}, false
	}
	r := rand.Intn(total)
	cum := 0
	var chosen string
	for _, e := range entries {
		cum += e.weight
		if r < cum {
			chosen = e.uci
			break
		}
	}
	if chosen == "" {
		chosen = entries[len(entries)-1].uci
	}
	m, ok := parseUCIMove(chosen, pos)
	if !ok {
		fmt.Printf("info string book move %s illegal in position\n", chosen)
	}
	return m, ok
}

// ---------------------------------------------------------------------------
// UCI main loop
// ---------------------------------------------------------------------------

func main() {
	rand.Seed(time.Now().UnixNano())
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

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if engine.handleCommand(line) {
			break
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
			fmt.Println("info string book move")
			fmt.Printf("bestmove %s\n", m.UCI())
			return
		}
		fmt.Println("info string book miss")
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
			if piece == king {
				p.kingSq[whiteIdx] = sq
			} else if piece == -king {
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
// Move generation
// ---------------------------------------------------------------------------

func (p *Position) generateLegalTo(list *MoveList, capturesOnly bool) {
	list.reset()
	var pseudo MoveList
	p.generatePseudoTo(&pseudo, capturesOnly)
	for i := 0; i < pseudo.count; i++ {
		m := pseudo.moves[i]
		u := p.MakeMove(m)
		legal := !p.inCheck(-p.sideToMove)
		p.UndoMove(m, u)
		if legal {
			list.add(m)
		}
	}
}

func (p *Position) generatePseudoTo(moves *MoveList, capturesOnly bool) {
	moves.reset()
	sideIdx := blackIdx
	if p.sideToMove == white {
		sideIdx = whiteIdx
	}
	occ := p.byColor[sideIdx]
	for occ != 0 {
		sq := bits.TrailingZeros64(occ)
		occ &= occ - 1
		switch abs(p.board[sq]) {
		case pawn:
			p.genPawnMoves(sq, capturesOnly, moves)
		case knight:
			p.genJumpMoves(sq, knightMask[sq], capturesOnly, moves)
		case bishop:
			p.genSliderMoves(sq, bishopDirs[:], capturesOnly, moves)
		case rook:
			p.genSliderMoves(sq, rookDirs[:], capturesOnly, moves)
		case queen:
			p.genSliderMoves(sq, queenDirs[:], capturesOnly, moves)
		case king:
			p.genJumpMoves(sq, kingMask[sq], capturesOnly, moves)
		}
	}
	if !capturesOnly {
		p.genCastling(moves)
	}
}

func (p *Position) genJumpMoves(sq int, mask uint64, capturesOnly bool, moves *MoveList) {
	sideIdx := blackIdx
	if p.sideToMove == white {
		sideIdx = whiteIdx
	}
	quiet := mask &^ p.byColor[0] &^ p.byColor[1]
	if !capturesOnly {
		for quiet != 0 {
			to := bits.TrailingZeros64(quiet)
			quiet &= quiet - 1
			moves.add(Move{From: sq, To: to})
		}
	}
	caps := mask & p.byColor[1-sideIdx]
	for caps != 0 {
		to := bits.TrailingZeros64(caps)
		caps &= caps - 1
		moves.add(Move{From: sq, To: to, Captured: p.board[to]})
	}
}

func (p *Position) genSliderMoves(sq int, dirs []int, capturesOnly bool, moves *MoveList) {
	for _, d := range dirs {
		to := sq
		for {
			next := to + d
			if !inBoard(next) || !sameLine(to, next, d) {
				break
			}
			to = next
			target := p.board[to]
			if target == empty {
				if !capturesOnly {
					moves.add(Move{From: sq, To: to})
				}
				continue
			}
			if colorOf(target) != p.sideToMove {
				moves.add(Move{From: sq, To: to, Captured: target})
			}
			break
		}
	}
}

func (p *Position) genPawnMoves(sq int, capturesOnly bool, moves *MoveList) {
	rank := sq / 8
	file := sq % 8
	dir := 8
	startRank := 1
	promoRank := 6
	colorIdx := whiteIdx
	if p.sideToMove == black {
		dir = -8
		startRank = 6
		promoRank = 1
		colorIdx = blackIdx
	}

	if !capturesOnly {
		one := sq + dir
		if inBoard(one) && p.board[one] == empty {
			if rank == promoRank {
				for _, promo := range promoPieces {
					moves.add(Move{From: sq, To: one, Promo: promo})
				}
			} else {
				moves.add(Move{From: sq, To: one})
				if rank == startRank {
					two := sq + 2*dir
					if inBoard(two) && p.board[two] == empty {
						moves.add(Move{From: sq, To: two})
					}
				}
			}
		}
	}

	caps := pawnMask[colorIdx][sq] & p.byColor[1-colorIdx]
	for caps != 0 {
		to := bits.TrailingZeros64(caps)
		caps &= caps - 1
		if rank == promoRank {
			for _, promo := range promoPieces {
				moves.add(Move{From: sq, To: to, Promo: promo, Captured: p.board[to]})
			}
		} else {
			moves.add(Move{From: sq, To: to, Captured: p.board[to]})
		}
	}
	if p.epSquare >= 0 {
		epBB := uint64(1) << uint(p.epSquare)
		if pawnMask[colorIdx][sq]&epBB != 0 {
			moves.add(Move{From: sq, To: p.epSquare, EnPassant: true,
				Captured: -p.sideToMove * pawn})
		}
	}
	_ = file
}

func (p *Position) genCastling(moves *MoveList) {
	if p.sideToMove == white && p.kingSq[whiteIdx] == 4 {
		if p.castlingRights&castleWK != 0 && p.board[5] == empty && p.board[6] == empty &&
			!p.squareAttacked(4, black) && !p.squareAttacked(5, black) && !p.squareAttacked(6, black) {
			moves.add(Move{From: 4, To: 6, Castle: true})
		}
		if p.castlingRights&castleWQ != 0 && p.board[3] == empty && p.board[2] == empty && p.board[1] == empty &&
			!p.squareAttacked(4, black) && !p.squareAttacked(3, black) && !p.squareAttacked(2, black) {
			moves.add(Move{From: 4, To: 2, Castle: true})
		}
	}
	if p.sideToMove == black && p.kingSq[blackIdx] == 60 {
		if p.castlingRights&castleBK != 0 && p.board[61] == empty && p.board[62] == empty &&
			!p.squareAttacked(60, white) && !p.squareAttacked(61, white) && !p.squareAttacked(62, white) {
			moves.add(Move{From: 60, To: 62, Castle: true})
		}
		if p.castlingRights&castleBQ != 0 && p.board[59] == empty && p.board[58] == empty && p.board[57] == empty &&
			!p.squareAttacked(60, white) && !p.squareAttacked(59, white) && !p.squareAttacked(58, white) {
			moves.add(Move{From: 60, To: 58, Castle: true})
		}
	}
}

func (p *Position) inCheck(color int) bool {
	idx := blackIdx
	if color == white {
		idx = whiteIdx
	}
	if p.kingSq[idx] < 0 {
		return false
	}
	return p.squareAttacked(p.kingSq[idx], -color)
}

func (p *Position) squareAttacked(sq int, byColor int) bool {
	byIdx := blackIdx
	if byColor == white {
		byIdx = whiteIdx
	}
	if p.byType[byIdx][knight]&knightMask[sq] != 0 ||
		p.byType[byIdx][king]&kingMask[sq] != 0 {
		return true
	}
	// pawn attacks: a pawn of byColor standing on a square that attacks sq
	if p.byType[byIdx][pawn]&pawnAttackers[byIdx][sq] != 0 {
		return true
	}
	for _, d := range bishopDirs {
		cur := sq
		for {
			next := cur + d
			if !inBoard(next) || !sameLine(cur, next, d) {
				break
			}
			cur = next
			piece := p.board[cur]
			if piece == empty {
				continue
			}
			if piece == byColor*bishop || piece == byColor*queen {
				return true
			}
			break
		}
	}
	for _, d := range rookDirs {
		cur := sq
		for {
			next := cur + d
			if !inBoard(next) || !sameLine(cur, next, d) {
				break
			}
			cur = next
			piece := p.board[cur]
			if piece == empty {
				continue
			}
			if piece == byColor*rook || piece == byColor*queen {
				return true
			}
			break
		}
	}
	return false
}

func (p *Position) givesCheck(m Move) bool {
	u := p.MakeMove(m)
	inChk := p.inCheck(p.sideToMove)
	p.UndoMove(m, u)
	return inChk
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

func evaluate(p *Position) int {
	phase := p.phase
	if phase > 24 {
		phase = 24
	}
	score := (p.mg*phase + p.eg*(24-phase)) / 24
	score += colorExtras(p, whiteIdx) - colorExtras(p, blackIdx)
	// tempo
	if p.sideToMove == white {
		score += 12
	} else {
		score -= 12
	}
	if p.sideToMove == black {
		return -score
	}
	return score
}

// colorExtras returns white-positive extras for one color (absolute terms).
func colorExtras(p *Position, ci int) int {
	opp := 1 - ci
	sign := 1
	if ci == blackIdx {
		sign = -1
	}
	bonus := 0

	ownBishops := p.byType[ci][bishop]
	if bits.OnesCount64(ownBishops) >= 2 {
		bonus += 28
	}

	ownPawns := p.byType[ci][pawn]
	oppPawns := p.byType[opp][pawn]
	for f := 0; f < 8; f++ {
		cnt := bits.OnesCount64(ownPawns & fileMask[f])
		if cnt > 1 {
			bonus -= 12 * (cnt - 1) // doubled
		}
		if cnt > 0 && ownPawns&adjFileMask[f] == 0 {
			bonus -= 14 // isolated
		}
	}
	// passed pawns
	bb := ownPawns
	for bb != 0 {
		sq := bits.TrailingZeros64(bb)
		bb &= bb - 1
		if oppPawns&passedMask[ci][sq] == 0 {
			rank := sq / 8
			if ci == blackIdx {
				rank = 7 - rank
			}
			bonus += passedBonus[rank]
		}
	}
	// rooks on open / semi-open files
	bb = p.byType[ci][rook]
	for bb != 0 {
		sq := bits.TrailingZeros64(bb)
		bb &= bb - 1
		f := sq % 8
		if ownPawns&fileMask[f] == 0 {
			if oppPawns&fileMask[f] == 0 {
				bonus += 24 // open
			} else {
				bonus += 10 // semi-open
			}
		}
	}
	// knight mobility (cheap)
	bb = p.byType[ci][knight]
	for bb != 0 {
		sq := bits.TrailingZeros64(bb)
		bb &= bb - 1
		bonus += 4 * (bits.OnesCount64(knightMask[sq]&^p.byColor[ci]) - 4)
	}
	// King safety: penalty for open/semi-open files near the king
	if p.kingSq[ci] >= 0 {
		kf := p.kingSq[ci] % 8
		openPen := 0
		for d := -1; d <= 1; d++ {
			ff := kf + d
			if ff < 0 || ff > 7 {
				continue
			}
			ownP := p.byType[ci][pawn]&fileMask[ff] != 0
			oppP := p.byType[1-ci][pawn]&fileMask[ff] != 0
			if !ownP {
				if !oppP {
					openPen += 18
				} else {
					openPen += 6
				}
			}
		}
		bonus -= openPen
	}
	_ = sign
	return bonus
}

// ---------------------------------------------------------------------------
// Time management
// ---------------------------------------------------------------------------

type timePlan struct {
	soft time.Time
	hard time.Time
}

func computeTimePlan(side int, limits SearchLimits) timePlan {
	now := time.Now()
	if limits.Infinite {
		return timePlan{}
	}
	if limits.MoveTime > 0 {
		hard := now.Add(limits.MoveTime * 95 / 100)
		return timePlan{soft: hard, hard: hard}
	}
	rem := limits.WTime
	inc := limits.WInc
	if side == black {
		rem = limits.BTime
		inc = limits.BInc
	}
	if rem <= 0 {
		return timePlan{}
	}
	// Emergency mode with very little time left.
	if rem < 200*time.Millisecond {
		return timePlan{
			soft: now.Add(rem / 4),
			hard: now.Add(rem * 3 / 4),
		}
	}
	budget := rem/22 + inc*4/5
	hard := budget * 5
	if hard > rem/3 {
		hard = rem / 3
	}
	if hard < budget {
		hard = budget
	}
	if hard > rem-100*time.Millisecond {
		hard = rem - 100*time.Millisecond
	}
	return timePlan{soft: now.Add(budget), hard: now.Add(hard)}
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

func SearchBestMove(ctx context.Context, pos Position, limits SearchLimits, hashMB int) Move {
	if limits.Threads < 1 {
		limits.Threads = 1
	}
	tt := newSharedTT(hashMB)
	gen := tt.nextGen()

	if limits.Threads == 1 {
		s := newSearcher(tt, gen, ctx, pos, limits)
		return s.searchRootMoves(pos, limits)
	}

	if !limits.Silent {
		fmt.Printf("info string Lazy SMP workers %d\n", limits.Threads)
	}
	type result struct {
		move  Move
		score int
		depth int
	}
	results := make(chan result, limits.Threads)
	var wg sync.WaitGroup
	for worker := 0; worker < limits.Threads; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			wl := limits
			wl.Silent = worker != 0
			s := newSearcher(tt, gen, ctx, pos, wl)
			move := s.searchRootMoves(pos, wl)
			results <- result{move: move, score: s.lastScore, depth: s.lastDepth}
		}(worker)
	}
	wg.Wait()
	close(results)
	best := result{move: Move{From: -1, To: -1}, depth: -1}
	for r := range results {
		if r.move.From >= 0 && (best.move.From < 0 || r.depth > best.depth ||
			(r.depth == best.depth && r.score > best.score)) {
			best = r
		}
	}
	return best.move
}

func newSearcher(tt *sharedTT, gen uint8, ctx context.Context, root Position, limits SearchLimits) *Searcher {
	plan := computeTimePlan(root.sideToMove, limits)
	s := &Searcher{tt: tt, gen: gen, ctx: ctx, start: time.Now(), softDead: plan.soft, hardDead: plan.hard}
	s.repeat[0] = root.hash
	s.repCount = 1
	return s
}

func (s *Searcher) stopRequested() bool {
	if s.stopped {
		return true
	}
	if s.ctx.Err() != nil {
		s.stopped = true
		return true
	}
	if !s.hardDead.IsZero() && time.Now().After(s.hardDead) {
		s.stopped = true
		return true
	}
	return false
}

func (s *Searcher) checkup() bool {
	if s.nodes&1023 == 0 && s.stopRequested() {
		return true
	}
	return false
}

func (s *Searcher) pushHash(hash uint64) {
	if s.repCount < len(s.repeat) {
		s.repeat[s.repCount] = hash
		s.repCount++
	}
}

func (s *Searcher) popHash() { s.repCount-- }

func (s *Searcher) isRepetition(hash uint64, halfmove int) bool {
	limit := s.repCount - 1 - halfmove
	if limit < 0 {
		limit = 0
	}
	for i := s.repCount - 2; i >= limit; i-- {
		if s.repeat[i] == hash {
			return true
		}
	}
	return false
}

func (s *Searcher) searchRootMoves(root Position, limits SearchLimits) Move {
	pos := root
	var rootMoves MoveList
	pos.generateLegalTo(&rootMoves, false)
	if rootMoves.count == 0 {
		return Move{From: -1, To: -1}
	}
	if rootMoves.count == 1 {
		if !limits.Silent {
			fmt.Printf("info depth 1 score cp 0 nodes 1\n")
		}
		s.lastScore = 0
		s.lastDepth = 1
		return rootMoves.moves[0]
	}

	maxDepth := limits.Depth
	if maxDepth <= 0 || maxDepth >= maxPly {
		maxDepth = maxPly - 8
	}

	ttMove, _ := s.probeMove(pos.hash)
	s.scoreMoves(&rootMoves, ttMove, 0, &pos, 0)

	bestMove := rootMoves.moves[0]
	bestScore := -mateScore
	prevIterElapsed := time.Duration(0)

	for depth := 1; depth <= maxDepth; depth++ {
		iterStart := time.Now()
		if depth > 1 {
			// Extrapolate: stop before starting an iteration we likely cannot finish.
			elapsed := iterStart.Sub(s.start)
			lastIter := iterStart.Sub(s.start) - prevIterElapsed
			if !s.softDead.IsZero() && lastIter > 0 {
				projected := elapsed + lastIter + lastIter/2
				if projected > s.softDead.Sub(s.start) {
					break
				}
			}
			if s.stopRequested() {
				break
			}
		}

		// aspiration window
		delta := 14
		if depth >= 5 {
			alpha := bestScore - delta
			beta := bestScore + delta
			var score, move int
			var iterBest Move
			var complete bool
			for {
				score, iterBest, complete = s.searchRootIter(&pos, depth, alpha, beta, bestMove)
				if !complete {
					return bestMove
				}
				if score <= alpha {
					alpha = max(-infinity, alpha-delta)
					delta += delta / 2
					continue
				}
				if score >= beta {
					beta = min(infinity, beta+delta)
					delta += delta / 2
					continue
				}
				move = score
				break
			}
			bestScore = move
			if iterBest.From >= 0 {
				bestMove = iterBest
			}
		} else {
			var iterBest Move
			score, iterBest, complete := s.searchRootIter(&pos, depth, -infinity, infinity, bestMove)
			if !complete {
				return bestMove
			}
			bestScore = score
			if iterBest.From >= 0 {
				bestMove = iterBest
			}
		}

		prevIterElapsed = iterStart.Sub(s.start)
		s.lastScore = bestScore
		s.lastDepth = depth
		s.pvLine = s.extractPV(&pos, depth)

		if !limits.Silent {
			elapsed := time.Since(s.start)
			nps := uint64(float64(s.nodes) / math.Max(0.001, elapsed.Seconds()))
			scoreStr := "cp"
			scoreVal := bestScore
			if bestScore > mateBound {
				scoreStr = "mate"
				scoreVal = (mateScore - bestScore + 1) / 2
			} else if bestScore < -mateBound {
				scoreStr = "mate"
				scoreVal = -((mateScore + bestScore + 1) / 2)
			}
			pvStr := ""
			for _, m := range s.pvLine {
				pvStr += " " + m.UCI()
			}
			if pvStr == "" {
				pvStr = " " + bestMove.UCI()
			}
			fmt.Printf("info depth %d score %s %d nodes %d nps %d time %d pv%s\n",
				depth, scoreStr, scoreVal, s.nodes, nps, elapsed.Milliseconds(), pvStr)
		}

		// stop early on a proven mate
		if bestScore > mateBound || bestScore < -mateBound {
			break
		}
	}
	return bestMove
}

func (s *Searcher) searchRootIter(pos *Position, depth, alpha, beta int, prevBest Move) (int, Move, bool) {
	var rootMoves MoveList
	pos.generateLegalTo(&rootMoves, false)
	if rootMoves.count == 0 {
		if pos.inCheck(pos.sideToMove) {
			return -mateScore + 0, Move{}, true
		}
		return 0, Move{}, true
	}
	ttMove, _ := s.probeMove(pos.hash)
	s.scoreMoves(&rootMoves, ttMove, 0, pos, 0)
	// put previous best move first
	for i := 0; i < rootMoves.count; i++ {
		if sameMove(rootMoves.moves[i], prevBest) && i > 0 {
			rootMoves.moves[i], rootMoves.moves[0] = rootMoves.moves[0], rootMoves.moves[i]
			rootMoves.scores[i], rootMoves.scores[0] = rootMoves.scores[0], rootMoves.scores[i]
			break
		}
	}

	bestScore := -infinity
	bestMove := prevBest
	origAlpha := alpha
	for i := 0; i < rootMoves.count; i++ {
		m := rootMoves.moves[i]
		// soft time check between root moves: keep searching only the first move
		if i > 0 && !s.softDead.IsZero() && time.Now().After(s.softDead) {
			// Out of soft time: abort this iteration and keep the previous best.
			return bestScore, bestMove, false
		}
		if s.stopRequested() {
			return bestScore, bestMove, false
		}
		u := pos.MakeMove(m)
		s.lastMoveTo[0] = m.To
		s.pushHash(pos.hash)
		var score int
		var complete bool
		if i == 0 {
			score, complete = s.alphaBeta(pos, depth-1, -beta, -alpha, 1, true, true)
			score = -score
		} else {
			score, complete = s.alphaBeta(pos, depth-1, -alpha-1, -alpha, 1, true, false)
			score = -score
			if complete && score > alpha && score < beta {
				score, complete = s.alphaBeta(pos, depth-1, -beta, -alpha, 1, true, true)
				score = -score
			}
		}
		s.popHash()
		pos.UndoMove(m, u)
		if !complete {
			return bestScore, bestMove, false
		}
		if score > bestScore {
			bestScore = score
			bestMove = m
		}
		if score > alpha {
			alpha = score
		}
		if alpha >= beta {
			break
		}
	}
	flag := boundType(bestScore, origAlpha, beta)
	s.storeTT(pos.hash, depth, bestScore, bestMove, flag, 0)
	return bestScore, bestMove, true
}

func (s *Searcher) alphaBeta(pos *Position, depth, alpha, beta, ply int, allowNull, isPV bool) (int, bool) {
	s.nodes++
	if s.checkup() {
		return 0, false
	}
	if ply >= maxPly-1 {
		return evaluate(pos), true
	}

	if pos.halfmove >= 100 || s.isRepetition(pos.hash, pos.halfmove) {
		return 0, true
	}
	if ply > 0 {
		if alpha < -mateBound {
			alpha = -mateBound
		}
		if alpha >= beta {
			return alpha, true
		}
	}

	inCheck := pos.inCheck(pos.sideToMove)
	if inCheck {
		depth++
	}

	// Mate distance pruning
	if ply > 0 {
		mateAlpha := -mateScore + ply
		if alpha < mateAlpha {
			alpha = mateAlpha
		}
		mateBeta := mateScore - ply - 1
		if beta > mateBeta {
			beta = mateBeta
		}
		if alpha >= beta {
			return alpha, true
		}
	}

	ttMove := Move{}
	if score, ok, move := s.probeTT(pos.hash, depth, alpha, beta, ply); ok {
		return score, true
	} else {
		ttMove = move
	}

	staticEval := evaluate(pos)

	if depth <= 0 {
		return s.quiescence(pos, alpha, beta, ply)
	}

	// Reverse futility pruning
	if !isPV && !inCheck && depth <= 6 && abs(beta) < mateBound {
		margin := 120 * depth
		if staticEval-margin >= beta {
			return staticEval, true
		}
	}

	// Razoring
	if !isPV && !inCheck && depth <= 3 && staticEval+150*depth < alpha {
		score, complete := s.quiescence(pos, alpha, beta, ply)
		if complete && score > alpha {
			return score, true
		}
	}

	// Null move pruning
	if allowNull && !isPV && depth >= 3 && !inCheck && hasNonPawnMaterial(pos, pos.sideToMove) && staticEval >= beta {
		r := 3 + depth/6
		u := pos.MakeNullMove()
		s.pushHash(pos.hash)
		nullScore, complete := s.alphaBeta(pos, depth-1-r, -beta, -beta+1, ply+1, false, false)
		s.popHash()
		pos.UndoNullMove(u)
		if !complete {
			return 0, false
		}
		if -nullScore >= beta {
			return -nullScore, true
		}
	}

	var moves MoveList
	pos.generateLegalTo(&moves, false)
	if moves.count == 0 {
		if inCheck {
			return -mateScore + ply, true
		}
		return 0, true
	}
	s.scoreMoves(&moves, ttMove, ply, pos, 0)

	bestScore := -infinity
	bestMove := Move{From: -1}
	origAlpha := alpha
	legalSearched := 0

	for i := 0; i < moves.count; i++ {
		m := moves.moves[i]
		isQuiet := m.Captured == empty && !m.EnPassant && m.Promo == empty

		// Futility pruning of quiet moves at shallow depth
		if !isPV && !inCheck && isQuiet && depth <= 6 && legalSearched > 0 &&
			staticEval+120*depth+100 <= alpha && abs(alpha) < mateBound {
			break
		}
		// Late Move Pruning: skip remaining quiet moves if many have been searched
		// and static eval is not close to alpha
		if !isPV && !inCheck && isQuiet && depth <= 8 && legalSearched >= (3+depth*depth/2) &&
			staticEval+80*depth < alpha && abs(alpha) < mateBound {
			break
		}

		u := pos.MakeMove(m)
		s.lastMoveTo[ply] = m.To
		s.pushHash(pos.hash)
		var score int
		var complete bool
		// LMR
		reduction := 0
		if depth >= 3 && legalSearched >= 3 && isQuiet && !inCheck {
			reduction = lmrTable[min(depth, 63)][min(legalSearched, 63)]
			if isPV {
				reduction--
			}
			if s.isKiller(m, ply) {
				reduction--
			}
			if reduction < 0 {
				reduction = 0
			}
			if reduction >= depth-1 {
				reduction = depth - 2
			}
		}
		if legalSearched == 0 {
			score, complete = s.alphaBeta(pos, depth-1, -beta, -alpha, ply+1, true, isPV)
			score = -score
		} else {
			score, complete = s.alphaBeta(pos, depth-1-reduction, -alpha-1, -alpha, ply+1, true, false)
			score = -score
			if complete && score > alpha && reduction > 0 {
				score, complete = s.alphaBeta(pos, depth-1, -alpha-1, -alpha, ply+1, true, false)
				score = -score
			}
			if complete && score > alpha && score < beta {
				score, complete = s.alphaBeta(pos, depth-1, -beta, -alpha, ply+1, true, isPV)
				score = -score
			}
		}
		s.popHash()
		pos.UndoMove(m, u)
		if !complete {
			return 0, false
		}
		legalSearched++

		if score > bestScore {
			bestScore = score
			bestMove = m
		}
		if score > alpha {
			alpha = score
		}
		if alpha >= beta {
			if isQuiet {
				s.recordCutoffs(pos, m, depth, ply)
			}
			break
		}
	}
	s.storeTT(pos.hash, depth, bestScore, bestMove, boundType(bestScore, origAlpha, beta), ply)
	return bestScore, true
}

func (s *Searcher) quiescence(pos *Position, alpha, beta, ply int) (int, bool) {
	s.nodes++
	if s.checkup() {
		return 0, false
	}
	if ply >= maxPly-1 {
		return evaluate(pos), true
	}
	standPat := evaluate(pos)
	if standPat >= beta {
		return standPat, true
	}
	// delta pruning
	bigDelta := 950
	if standPat+bigDelta < alpha {
		return alpha, true
	}
	if standPat > alpha {
		alpha = standPat
	}

	var caps MoveList
	pos.generateLegalTo(&caps, true)
	s.scoreMoves(&caps, Move{}, ply, pos, 1)

	best := standPat
	for i := 0; i < caps.count; i++ {
		m := caps.moves[i]
		// MVV/delta pruning per capture
		gain := egValue[abs(m.Captured)]
		if m.Promo != empty {
			gain += egValue[m.Promo] - egValue[pawn]
		}
		if standPat+gain+150 < alpha {
			continue
		}
		u := pos.MakeMove(m)
		s.pushHash(pos.hash)
		score, complete := s.quiescence(pos, -beta, -alpha, ply+1)
		s.popHash()
		pos.UndoMove(m, u)
		if !complete {
			return 0, false
		}
		score = -score
		if score > best {
			best = score
		}
		if score > alpha {
			alpha = score
		}
		if alpha >= beta {
			return score, true
		}
	}
	return best, true
}

func (s *Searcher) isKiller(m Move, ply int) bool {
	if ply >= maxPly {
		return false
	}
	return sameMove(m, s.killers[ply][0]) || sameMove(m, s.killers[ply][1])
}

func (s *Searcher) recordCutoffs(pos *Position, m Move, depth, ply int) {
	if ply < maxPly {
		s.killers[ply][1] = s.killers[ply][0]
		s.killers[ply][0] = m
	}
	sideIdx := blackIdx
	if pos.sideToMove == white {
		sideIdx = whiteIdx
	}
	s.bumpHistory(sideIdx, m, depth)
	if ply > 0 {
		prev := s.lastMoveTo[ply-1]
		if prev >= 0 {
			s.counter[sideIdx][prev][m.To] = m
		}
	}
}

func (s *Searcher) bumpHistory(sideIdx int, m Move, depth int) {
	h := &s.history[sideIdx][m.From][m.To]
	*h += depth * depth
	if *h > maxHistory {
		*h = maxHistory
	}
}

func (s *Searcher) scoreMoves(moves *MoveList, ttMove Move, ply int, pos *Position, capsOnly int) {
	sideIdx := blackIdx
	if pos.sideToMove == white {
		sideIdx = whiteIdx
	}
	var counterFrom = -1
	if ply > 0 && ply < maxPly {
		counterFrom = s.lastMoveTo[ply-1]
	}
	for i := 0; i < moves.count; i++ {
		m := moves.moves[i]
		score := 0
		if sameMove(m, ttMove) {
			score = 1 << 30
		} else if m.Captured != empty || m.EnPassant {
			victim := egValue[abs(m.Captured)]
			attacker := egValue[abs(pos.board[m.From])]
			if m.EnPassant {
				attacker = egValue[pawn]
				victim = egValue[pawn]
			}
			score = 16_000_000 + victim*16 - attacker
			if m.Promo != empty {
				score += egValue[m.Promo]
			}
		} else {
			if ply < maxPly {
				if sameMove(m, s.killers[ply][0]) {
					score = 8_000_000
				} else if sameMove(m, s.killers[ply][1]) {
					score = 7_900_000
				}
			}
			if counterFrom >= 0 && sameMove(m, s.counter[sideIdx][counterFrom][m.To]) {
				score = 7_500_000
			}
			if score < 7_000_000 {
				score = min(s.history[sideIdx][m.From][m.To], 6_999_999)
			}
		}
		moves.scores[i] = score
	}
	// selection sort on demand is replaced by full insertion sort here (move
	// lists are short and mostly ordered after the first few plies)
	for i := 1; i < moves.count; i++ {
		ms := moves.scores[i]
		mv := moves.moves[i]
		j := i - 1
		for j >= 0 && moves.scores[j] < ms {
			moves.scores[j+1] = moves.scores[j]
			moves.moves[j+1] = moves.moves[j]
			j--
		}
		moves.scores[j+1] = ms
		moves.moves[j+1] = mv
	}
}

func (s *Searcher) probeMove(hash uint64) (Move, bool) {
	sh := s.tt.shard(hash)
	sh.mu.RLock()
	entry := sh.entries[int(hash)&(len(sh.entries)-1)]
	sh.mu.RUnlock()
	if entry.key != hash {
		return Move{}, false
	}
	return entry.move, true
}

func (s *Searcher) extractPV(pos *Position, depth int) []Move {
	pv := make([]Move, 0, depth+4)
	seen := make(map[uint64]bool)
	cp := *pos
	for d := 0; d <= depth+4 && len(pv) < 32; d++ {
		m, ok := s.probeMove(cp.hash)
		if !ok || m.From < 0 || seen[cp.hash] {
			break
		}
		seen[cp.hash] = true
		legal := false
		var list MoveList
		cp.generateLegalTo(&list, false)
		for i := 0; i < list.count; i++ {
			if sameMove(list.moves[i], m) {
				legal = true
				break
			}
		}
		if !legal {
			break
		}
		cp.MakeMove(m)
		pv = append(pv, m)
		if len(pv) > depth+4 {
			break
		}
	}
	// restore not needed; cp is a copy
	return pv
}

func boundType(score, alpha, beta int) uint8 {
	if score <= alpha {
		return ttFlagUpper
	}
	if score >= beta {
		return ttFlagLower
	}
	return ttFlagExact
}

func hasNonPawnMaterial(pos *Position, side int) bool {
	ci := blackIdx
	if side == white {
		ci = whiteIdx
	}
	return pos.byType[ci][knight] != 0 || pos.byType[ci][bishop] != 0 ||
		pos.byType[ci][rook] != 0 || pos.byType[ci][queen] != 0
}

func (p *Position) computeHash() uint64 {
	var h uint64
	for sq, piece := range p.board {
		if piece != empty {
			h ^= zobristPiece[pieceHashIdx(piece)][sq]
		}
	}
	h ^= zobristCastling[p.castlingRights] ^ zobristEPFile[epFileIndex(p.epSquare)]
	if p.sideToMove == black {
		h ^= zobristSide
	}
	return h
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

func perft(pos *Position, depth int) int64 {
	if depth == 0 {
		return 1
	}
	var moves MoveList
	pos.generateLegalTo(&moves, false)
	if depth == 1 {
		return int64(moves.count)
	}
	var total int64
	for i := 0; i < moves.count; i++ {
		m := moves.moves[i]
		u := pos.MakeMove(m)
		total += perft(pos, depth-1)
		pos.UndoMove(m, u)
	}
	return total
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
