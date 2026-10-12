package main

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
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

	castleWK = 1 << 0
	castleWQ = 1 << 1
	castleBK = 1 << 2
	castleBQ = 1 << 3

	mateScore = 100000
	maxPly    = 64

	ttFlagExact = 0
	ttFlagLower = 1
	ttFlagUpper = 2

	maxHistoryScore = 16384
)

type Move struct {
	From      int
	To        int
	Promo     int
	EnPassant bool
	Castle    bool
	Captured  int
}

type Undo struct {
	captured       int
	castlingRights int
	epSquare       int
	halfmove       int
	fullmove       int
	hash           uint64
	kingSq         [2]int
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
}

type SearchLimits struct {
	Depth    int
	MoveTime time.Duration
	Infinite bool
	WTime    time.Duration
	BTime    time.Duration
	Silent   bool
}

type Engine struct {
	mu     sync.Mutex
	pos    Position
	cancel context.CancelFunc
	busy   bool
	wg     sync.WaitGroup
}

type MoveList struct {
	moves [256]Move
	count int
}

func (ml *MoveList) reset() {
	ml.count = 0
}

func (ml *MoveList) add(m Move) {
	ml.moves[ml.count] = m
	ml.count++
}

type TTEntry struct {
	key   uint64
	move  Move
	score int
	depth int8
	flag  uint8
	age   uint8
}

type Searcher struct {
	tt       []TTEntry
	age      uint8
	nodes    uint64
	history  [2][64][64]int
	killers  [maxPly][2]Move
	repeat   [maxPly + 256]uint64
	repCount int
}

var globalSearcher = newSearcher(1 << 20)

var (
	knightOffsets = [8]int{17, 15, 10, 6, -17, -15, -10, -6}
	bishopDirs    = [4]int{9, 7, -9, -7}
	rookDirs      = [4]int{8, -8, 1, -1}
	queenDirs     = [8]int{9, 7, -9, -7, 8, -8, 1, -1}
	kingOffsets   = [8]int{8, -8, 1, -1, 9, 7, -9, -7}

	pieceValues = [7]int{0, 100, 320, 330, 500, 900, 0}
	promoPieces = [4]int{queen, rook, bishop, knight}

	knightAttacks [64][8]int
	knightCount   [64]int
	kingAttacks   [64][8]int
	kingCount     [64]int
	pawnAttacks   [2][64][2]int
	pawnCount     [2][64]int

	zobristPiece    [12][64]uint64
	zobristCastling [16]uint64
	zobristEPFile   [9]uint64
	zobristSide     uint64

	pstPawn = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		5, 10, 10, -20, -20, 10, 10, 5,
		5, -5, -10, 0, 0, -10, -5, 5,
		0, 0, 0, 20, 20, 0, 0, 0,
		5, 5, 10, 25, 25, 10, 5, 5,
		10, 10, 20, 30, 30, 20, 10, 10,
		50, 50, 50, 50, 50, 50, 50, 50,
		0, 0, 0, 0, 0, 0, 0, 0,
	}
	pstKnight = [64]int{
		-50, -40, -30, -30, -30, -30, -40, -50,
		-40, -20, 0, 0, 0, 0, -20, -40,
		-30, 0, 10, 15, 15, 10, 0, -30,
		-30, 5, 15, 20, 20, 15, 5, -30,
		-30, 0, 15, 20, 20, 15, 0, -30,
		-30, 5, 10, 15, 15, 10, 5, -30,
		-40, -20, 0, 5, 5, 0, -20, -40,
		-50, -40, -30, -30, -30, -30, -40, -50,
	}
	pstBishop = [64]int{
		-20, -10, -10, -10, -10, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 5, 10, 10, 5, 0, -10,
		-10, 5, 5, 10, 10, 5, 5, -10,
		-10, 0, 10, 10, 10, 10, 0, -10,
		-10, 10, 10, 10, 10, 10, 10, -10,
		-10, 5, 0, 0, 0, 0, 5, -10,
		-20, -10, -10, -10, -10, -10, -10, -20,
	}
	pstRook = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		5, 10, 10, 10, 10, 10, 10, 5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		0, 0, 0, 5, 5, 0, 0, 0,
	}
	pstQueen = [64]int{
		-20, -10, -10, -5, -5, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 5, 5, 5, 5, 0, -10,
		-5, 0, 5, 5, 5, 5, 0, -5,
		0, 0, 5, 5, 5, 5, 0, -5,
		-10, 5, 5, 5, 5, 5, 0, -10,
		-10, 0, 5, 0, 0, 0, 0, -10,
		-20, -10, -10, -5, -5, -10, -10, -20,
	}
	pstKing = [64]int{
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-20, -30, -30, -40, -40, -30, -30, -20,
		-10, -20, -20, -20, -20, -20, -20, -10,
		20, 20, 0, 0, 0, 0, 20, 20,
		20, 30, 10, 0, 0, 10, 30, 20,
	}
)

func init() {
	initAttackTables()
	initZobrist()
}

func initAttackTables() {
	for sq := 0; sq < 64; sq++ {
		for _, off := range knightOffsets {
			to := sq + off
			if inBoard(to) && sameKnightPattern(sq, to) {
				knightAttacks[sq][knightCount[sq]] = to
				knightCount[sq]++
			}
		}
		for _, off := range kingOffsets {
			to := sq + off
			if inBoard(to) && sameKingPattern(sq, to) {
				kingAttacks[sq][kingCount[sq]] = to
				kingCount[sq]++
			}
		}
		file := sq % 8
		if file > 0 {
			pawnAttacks[0][sq][pawnCount[0][sq]] = sq + 7
			pawnCount[0][sq]++
		}
		if file < 7 {
			pawnAttacks[0][sq][pawnCount[0][sq]] = sq + 9
			pawnCount[0][sq]++
		}
		if file > 0 {
			pawnAttacks[1][sq][pawnCount[1][sq]] = sq - 9
			pawnCount[1][sq]++
		}
		if file < 7 {
			pawnAttacks[1][sq][pawnCount[1][sq]] = sq - 7
			pawnCount[1][sq]++
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

func newSearcher(ttSize int) *Searcher {
	if ttSize < 1 {
		ttSize = 1 << 20
	}
	// Keep table size power of two for fast masking.
	size := 1
	for size < ttSize {
		size <<= 1
	}
	return &Searcher{tt: make([]TTEntry, size)}
}

func main() {
	engine := &Engine{pos: NewStartPosition()}
	scanner := bufio.NewScanner(os.Stdin)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if engine.handleCommand(line) {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "input error:", err)
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
		fmt.Println("id name Waldemar")
		fmt.Println("id author zlElo")
		fmt.Println("uciok")
	case "isready":
		fmt.Println("readyok")
	case "ucinewgame":
		e.stopSearch()
		e.mu.Lock()
		e.pos = NewStartPosition()
		e.mu.Unlock()
	case "position":
		e.stopSearch()
		e.mu.Lock()
		defer e.mu.Unlock()
		if err := e.applyPosition(fields[1:]); err != nil {
			fmt.Fprintf(os.Stderr, "position error: %v\n", err)
		}
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
		if len(fields) < 7 {
			return fmt.Errorf("invalid fen")
		}
		fen := strings.Join(fields[1:7], " ")
		pos, err := NewPositionFromFEN(fen)
		if err != nil {
			return err
		}
		e.pos = pos
		idx = 7
	default:
		return fmt.Errorf("unknown position mode")
	}

	if idx < len(fields) && fields[idx] == "moves" {
		for _, moveStr := range fields[idx+1:] {
			move, ok := parseUCIMove(moveStr, e.pos)
			if !ok {
				return fmt.Errorf("illegal move %s", moveStr)
			}
			e.pos.MakeMove(move)
		}
	}
	return nil
}

func parseGo(fields []string) SearchLimits {
	limits := SearchLimits{Depth: 64}
	for i := 0; i < len(fields); i++ {
		tok := fields[i]
		switch tok {
		case "depth":
			if i+1 < len(fields) {
				if v, err := strconv.Atoi(fields[i+1]); err == nil && v > 0 {
					limits.Depth = v
				}
				i++
			}
		case "movetime":
			if i+1 < len(fields) {
				if v, err := strconv.Atoi(fields[i+1]); err == nil && v > 0 {
					limits.MoveTime = time.Duration(v) * time.Millisecond
				}
				i++
			}
		case "infinite":
			limits.Infinite = true
			limits.Depth = 64
		case "wtime":
			if i+1 < len(fields) {
				if v, err := strconv.Atoi(fields[i+1]); err == nil && v > 0 {
					limits.WTime = time.Duration(v) * time.Millisecond
				}
				i++
			}
		case "btime":
			if i+1 < len(fields) {
				if v, err := strconv.Atoi(fields[i+1]); err == nil && v > 0 {
					limits.BTime = time.Duration(v) * time.Millisecond
				}
				i++
			}
		}
	}
	return limits
}

func (e *Engine) startSearch(l SearchLimits) {
	e.stopSearch()

	e.mu.Lock()
	pos := e.pos.Clone()
	e.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	e.mu.Lock()
	e.cancel = cancel
	e.busy = true
	e.mu.Unlock()

	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		best := SearchBestMove(ctx, pos, l)
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
	if len(parts) != 6 {
		return p, fmt.Errorf("invalid fen field count")
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
			if piece == 0 {
				return p, fmt.Errorf("invalid fen piece")
			}
			p.board[sq] = piece
			if piece == king {
				p.kingSq[0] = sq
			} else if piece == -king {
				p.kingSq[1] = sq
			}
			file++
		}
		if file != 8 {
			return p, fmt.Errorf("invalid fen file count")
		}
	}

	if parts[1] == "w" {
		p.sideToMove = white
	} else if parts[1] == "b" {
		p.sideToMove = black
	} else {
		return p, fmt.Errorf("invalid side to move")
	}

	if parts[2] != "-" {
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

	if parts[3] != "-" {
		sq, ok := squareFromString(parts[3])
		if !ok {
			return p, fmt.Errorf("invalid ep square")
		}
		p.epSquare = sq
	}

	if v, err := strconv.Atoi(parts[4]); err == nil {
		p.halfmove = v
	}
	if v, err := strconv.Atoi(parts[5]); err == nil {
		p.fullmove = v
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
		return 0
	}
}

func (p Position) Clone() Position {
	return p
}

func (p *Position) MakeMove(m Move) Undo {
	undo := Undo{
		captured:       p.board[m.To],
		castlingRights: p.castlingRights,
		epSquare:       p.epSquare,
		halfmove:       p.halfmove,
		fullmove:       p.fullmove,
		hash:           p.hash,
		kingSq:         p.kingSq,
	}
	piece := p.board[m.From]
	movingType := abs(piece)

	p.epSquare = -1
	if movingType == pawn || undo.captured != empty || m.EnPassant {
		p.halfmove = 0
	} else {
		p.halfmove++
	}
	if p.sideToMove == black {
		p.fullmove++
	}

	p.hash ^= zobristCastling[p.castlingRights]
	p.hash ^= zobristEPFile[epFileIndex(undo.epSquare)]
	p.hash ^= zobristPiece[pieceToHashIndex(piece)][m.From]
	p.board[m.From] = empty

	if m.EnPassant {
		if p.sideToMove == white {
			capSq := m.To - 8
			undo.captured = p.board[capSq]
			p.hash ^= zobristPiece[pieceToHashIndex(undo.captured)][capSq]
			p.board[capSq] = empty
		} else {
			capSq := m.To + 8
			undo.captured = p.board[capSq]
			p.hash ^= zobristPiece[pieceToHashIndex(undo.captured)][capSq]
			p.board[capSq] = empty
		}
	} else if undo.captured != empty {
		p.hash ^= zobristPiece[pieceToHashIndex(undo.captured)][m.To]
	}

	if m.Castle {
		switch m.To {
		case 6:
			p.hash ^= zobristPiece[pieceToHashIndex(p.board[7])][7]
			p.board[5] = p.board[7]
			p.board[7] = empty
			p.hash ^= zobristPiece[pieceToHashIndex(p.board[5])][5]
		case 2:
			p.hash ^= zobristPiece[pieceToHashIndex(p.board[0])][0]
			p.board[3] = p.board[0]
			p.board[0] = empty
			p.hash ^= zobristPiece[pieceToHashIndex(p.board[3])][3]
		case 62:
			p.hash ^= zobristPiece[pieceToHashIndex(p.board[63])][63]
			p.board[61] = p.board[63]
			p.board[63] = empty
			p.hash ^= zobristPiece[pieceToHashIndex(p.board[61])][61]
		case 58:
			p.hash ^= zobristPiece[pieceToHashIndex(p.board[56])][56]
			p.board[59] = p.board[56]
			p.board[56] = empty
			p.hash ^= zobristPiece[pieceToHashIndex(p.board[59])][59]
		}
	}

	if m.Promo != 0 {
		if piece > 0 {
			piece = m.Promo
		} else {
			piece = -m.Promo
		}
	}
	p.board[m.To] = piece
	p.hash ^= zobristPiece[pieceToHashIndex(piece)][m.To]

	if movingType == pawn && abs(m.To-m.From) == 16 {
		if p.sideToMove == white {
			p.epSquare = m.From + 8
		} else {
			p.epSquare = m.From - 8
		}
	}

	if movingType == king {
		if p.sideToMove == white {
			p.kingSq[0] = m.To
		} else {
			p.kingSq[1] = m.To
		}
	}

	p.updateCastlingRights(m, movingType, undo.captured)
	p.hash ^= zobristCastling[p.castlingRights]
	p.hash ^= zobristEPFile[epFileIndex(p.epSquare)]
	p.hash ^= zobristSide
	p.sideToMove = -p.sideToMove
	return undo
}

func (p *Position) UndoMove(m Move, u Undo) {
	p.sideToMove = -p.sideToMove
	p.castlingRights = u.castlingRights
	p.epSquare = u.epSquare
	p.halfmove = u.halfmove
	p.fullmove = u.fullmove
	p.hash = u.hash
	p.kingSq = u.kingSq

	piece := p.board[m.To]
	if m.Promo != 0 {
		if p.sideToMove == white {
			piece = pawn
		} else {
			piece = -pawn
		}
	}
	p.board[m.From] = piece
	p.board[m.To] = u.captured
	if m.EnPassant {
		p.board[m.To] = empty
		if p.sideToMove == white {
			p.board[m.To-8] = -pawn
		} else {
			p.board[m.To+8] = pawn
		}
	}
	if m.Castle {
		switch m.To {
		case 6:
			p.board[7] = p.board[5]
			p.board[5] = empty
		case 2:
			p.board[0] = p.board[3]
			p.board[3] = empty
		case 62:
			p.board[63] = p.board[61]
			p.board[61] = empty
		case 58:
			p.board[56] = p.board[59]
			p.board[59] = empty
		}
	}

}

func (p *Position) MakeNullMove() Undo {
	u := Undo{
		castlingRights: p.castlingRights,
		epSquare:       p.epSquare,
		halfmove:       p.halfmove,
		fullmove:       p.fullmove,
		hash:           p.hash,
		kingSq:         p.kingSq,
	}
	p.hash ^= zobristEPFile[epFileIndex(p.epSquare)]
	p.epSquare = -1
	p.hash ^= zobristEPFile[0]
	p.hash ^= zobristSide
	if p.sideToMove == black {
		p.fullmove++
	}
	p.halfmove++
	p.sideToMove = -p.sideToMove
	return u
}

func (p *Position) UndoNullMove(u Undo) {
	p.sideToMove = -p.sideToMove
	p.castlingRights = u.castlingRights
	p.epSquare = u.epSquare
	p.halfmove = u.halfmove
	p.fullmove = u.fullmove
	p.hash = u.hash
	p.kingSq = u.kingSq
}

func (p *Position) updateCastlingRights(m Move, movingType int, captured int) {
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

	if movingType == king {
		if p.sideToMove == white {
			p.castlingRights &^= castleWK | castleWQ
		} else {
			p.castlingRights &^= castleBK | castleBQ
		}
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

func (p Position) GenerateLegalMoves() []Move {
	cp := p
	var list MoveList
	cp.generateLegalMovesTo(&list, false)
	moves := make([]Move, list.count)
	copy(moves, list.moves[:list.count])
	return moves
}

func (p *Position) generateLegalMovesTo(legal *MoveList, capturesOnly bool) {
	legal.reset()
	var pseudo MoveList
	p.generatePseudoMovesTo(&pseudo, capturesOnly)
	for i := 0; i < pseudo.count; i++ {
		m := pseudo.moves[i]
		u := p.MakeMove(m)
		if !p.InCheck(-p.sideToMove) {
			legal.add(m)
		}
		p.UndoMove(m, u)
	}
}

func (p *Position) generatePseudoMovesTo(moves *MoveList, capturesOnly bool) {
	moves.reset()
	for sq, piece := range p.board {
		if piece == empty || colorOf(piece) != p.sideToMove {
			continue
		}
		switch abs(piece) {
		case pawn:
			p.generatePawnMoves(sq, capturesOnly, moves)
		case knight:
			p.generateKnightMoves(sq, capturesOnly, moves)
		case bishop:
			p.generateSliderMoves(sq, bishopDirs[:], capturesOnly, moves)
		case rook:
			p.generateSliderMoves(sq, rookDirs[:], capturesOnly, moves)
		case queen:
			p.generateSliderMoves(sq, queenDirs[:], capturesOnly, moves)
		case king:
			p.generateKingMoves(sq, capturesOnly, moves)
		}
	}
}

func (p *Position) generatePawnMoves(sq int, capturesOnly bool, moves *MoveList) {
	rank := sq / 8
	file := sq % 8
	dir := 8
	startRank := 1
	promoRank := 6
	enemy := black
	colorIdx := 0
	if p.sideToMove == black {
		dir = -8
		startRank = 6
		promoRank = 1
		enemy = white
		colorIdx = 1
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

	for i := 0; i < pawnCount[colorIdx][sq]; i++ {
		to := pawnAttacks[colorIdx][sq][i]
		if !inBoard(to) || abs((to%8)-file) != 1 {
			continue
		}
		target := p.board[to]
		if target != empty && colorOf(target) == enemy {
			if rank == promoRank {
				for _, promo := range promoPieces {
					moves.add(Move{From: sq, To: to, Promo: promo, Captured: target})
				}
			} else {
				moves.add(Move{From: sq, To: to, Captured: target})
			}
		}
		if to == p.epSquare {
			moves.add(Move{From: sq, To: to, EnPassant: true, Captured: -p.sideToMove * pawn})
		}
	}
}

func (p *Position) generateKnightMoves(sq int, capturesOnly bool, moves *MoveList) {
	for i := 0; i < knightCount[sq]; i++ {
		to := knightAttacks[sq][i]
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
	}
}

func (p *Position) generateSliderMoves(sq int, dirs []int, capturesOnly bool, moves *MoveList) {
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

func (p *Position) generateKingMoves(sq int, capturesOnly bool, moves *MoveList) {
	for i := 0; i < kingCount[sq]; i++ {
		to := kingAttacks[sq][i]
		target := p.board[to]
		if target == empty {
			if !capturesOnly {
				moves.add(Move{From: sq, To: to})
			}
		} else if colorOf(target) != p.sideToMove {
			moves.add(Move{From: sq, To: to, Captured: target})
		}
	}
	if capturesOnly {
		return
	}
	if p.sideToMove == white && sq == 4 {
		if p.castlingRights&castleWK != 0 && p.board[5] == empty && p.board[6] == empty && !p.InCheck(white) && !p.squareAttacked(5, black) && !p.squareAttacked(6, black) {
			moves.add(Move{From: 4, To: 6, Castle: true})
		}
		if p.castlingRights&castleWQ != 0 && p.board[3] == empty && p.board[2] == empty && p.board[1] == empty && !p.InCheck(white) && !p.squareAttacked(3, black) && !p.squareAttacked(2, black) {
			moves.add(Move{From: 4, To: 2, Castle: true})
		}
	}
	if p.sideToMove == black && sq == 60 {
		if p.castlingRights&castleBK != 0 && p.board[61] == empty && p.board[62] == empty && !p.InCheck(black) && !p.squareAttacked(61, white) && !p.squareAttacked(62, white) {
			moves.add(Move{From: 60, To: 62, Castle: true})
		}
		if p.castlingRights&castleBQ != 0 && p.board[59] == empty && p.board[58] == empty && p.board[57] == empty && !p.InCheck(black) && !p.squareAttacked(59, white) && !p.squareAttacked(58, white) {
			moves.add(Move{From: 60, To: 58, Castle: true})
		}
	}
}

func (p Position) InCheck(color int) bool {
	idx := 0
	if color == black {
		idx = 1
	}
	kingSq := p.kingSq[idx]
	if kingSq < 0 {
		return false
	}
	return p.squareAttacked(kingSq, -color)
}

func (p Position) squareAttacked(sq int, byColor int) bool {
	if byColor == white {
		for i := 0; i < pawnCount[1][sq]; i++ {
			from := pawnAttacks[1][sq][i]
			if inBoard(from) && p.board[from] == pawn {
				return true
			}
		}
	} else {
		for i := 0; i < pawnCount[0][sq]; i++ {
			from := pawnAttacks[0][sq][i]
			if inBoard(from) && p.board[from] == -pawn {
				return true
			}
		}
	}

	for i := 0; i < knightCount[sq]; i++ {
		from := knightAttacks[sq][i]
		if p.board[from] == byColor*knight {
			return true
		}
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

	for i := 0; i < kingCount[sq]; i++ {
		from := kingAttacks[sq][i]
		if p.board[from] == byColor*king {
			return true
		}
	}
	return false
}

func SearchBestMove(ctx context.Context, pos Position, limits SearchLimits) Move {
	return globalSearcher.searchBestMove(ctx, pos, limits)
}

func (s *Searcher) searchBestMove(ctx context.Context, pos Position, limits SearchLimits) Move {
	var rootMoves MoveList
	pos.generateLegalMovesTo(&rootMoves, false)
	if rootMoves.count == 0 {
		return Move{From: -1, To: -1}
	}

	maxDepth := limits.Depth
	if maxDepth <= 0 {
		maxDepth = 64
	}
	deadline := computeDeadline(pos.sideToMove, limits)
	s.nodes = 0
	s.age++
	s.repCount = 0
	s.repeat[s.repCount] = pos.hash
	s.repCount++

	bestMove := rootMoves.moves[0]
	bestScore := -mateScore
	aspiration := 30
	alpha := -mateScore
	beta := mateScore
	start := time.Now()

	for depth := 1; depth <= maxDepth; depth++ {
		if searchStopped(ctx, deadline) {
			break
		}
		if depth > 2 {
			alpha = bestScore - aspiration
			beta = bestScore + aspiration
		} else {
			alpha, beta = -mateScore, mateScore
		}

		var score int
		var complete bool
		for {
			score, bestMove, complete = s.searchRoot(ctx, &pos, depth, alpha, beta, deadline)
			if !complete {
				return bestMove
			}
			if score <= alpha && depth > 2 {
				alpha -= aspiration
				aspiration *= 2
				continue
			}
			if score >= beta && depth > 2 {
				beta += aspiration
				aspiration *= 2
				continue
			}
			break
		}

		bestScore = score
		if !limits.Silent {
			nps := int(float64(s.nodes) / math.Max(0.001, time.Since(start).Seconds()))
			fmt.Printf("info depth %d score cp %d nodes %d nps %d pv %s\n", depth, bestScore, s.nodes, nps, bestMove.UCI())
		}
	}
	return bestMove
}

func (s *Searcher) searchRoot(ctx context.Context, pos *Position, depth, alpha, beta int, deadline time.Time) (int, Move, bool) {
	var legal MoveList
	pos.generateLegalMovesTo(&legal, false)
	if legal.count == 0 {
		return terminalScore(*pos, depth), Move{From: -1, To: -1}, true
	}
	ttMove, _ := s.probeMove(pos.hash)
	s.orderMoves(&legal, ttMove, 0, pos.sideToMove)

	bestMove := legal.moves[0]
	bestScore := -mateScore
	origAlpha := alpha
	for i := 0; i < legal.count; i++ {
		if s.shouldStop(ctx, deadline) {
			return bestScore, bestMove, false
		}
		m := legal.moves[i]
		undo := pos.MakeMove(m)
		s.pushHash(pos.hash)
		var score int
		var complete bool
		if i == 0 {
			score, complete = s.alphaBeta(ctx, pos, depth-1, -beta, -alpha, 1, true, deadline)
			score = -score
		} else {
			score, complete = s.alphaBeta(ctx, pos, depth-1, -alpha-1, -alpha, 1, true, deadline)
			score = -score
			if complete && score > alpha && score < beta {
				score, complete = s.alphaBeta(ctx, pos, depth-1, -beta, -alpha, 1, true, deadline)
				score = -score
			}
		}
		s.popHash()
		pos.UndoMove(m, undo)
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
	s.storeTT(pos.hash, depth, bestScore, bestMove, boundType(bestScore, origAlpha, beta))
	return bestScore, bestMove, true
}

func (s *Searcher) alphaBeta(ctx context.Context, pos *Position, depth, alpha, beta, ply int, allowNull bool, deadline time.Time) (int, bool) {
	s.nodes++
	if s.nodes&2047 == 0 && s.shouldStop(ctx, deadline) {
		return 0, false
	}
	if ply >= maxPly-1 {
		return evaluate(*pos), true
	}

	if pos.halfmove >= 100 || s.isRepetition(pos.hash) {
		return 0, true
	}
	inCheck := pos.InCheck(pos.sideToMove)
	if depth <= 0 {
		return s.quiescence(ctx, pos, alpha, beta, ply, deadline)
	}

	origAlpha := alpha
	ttMove := Move{}
	if score, ok, move := s.probeTT(pos.hash, depth, alpha, beta, ply); ok {
		return score, true
	} else {
		ttMove = move
	}

	staticEval := evaluate(*pos)
	if !inCheck && depth <= 2 {
		margin := 100 + depth*80
		if staticEval-margin >= beta {
			return staticEval, true
		}
		if depth == 1 && staticEval+120 <= alpha {
			return staticEval, true
		}
	}

	if allowNull && depth >= 3 && !inCheck && hasNonPawnMaterial(*pos, pos.sideToMove) {
		u := pos.MakeNullMove()
		s.pushHash(pos.hash)
		nullScore, complete := s.alphaBeta(ctx, pos, depth-1-2, -beta, -beta+1, ply+1, false, deadline)
		s.popHash()
		pos.UndoNullMove(u)
		if !complete {
			return 0, false
		}
		nullScore = -nullScore
		if nullScore >= beta {
			return nullScore, true
		}
	}

	var legal MoveList
	pos.generateLegalMovesTo(&legal, false)
	if legal.count == 0 {
		return terminalScore(*pos, depth), true
	}
	s.orderMoves(&legal, ttMove, ply, pos.sideToMove)

	bestScore := -mateScore
	bestMove := legal.moves[0]
	for i := 0; i < legal.count; i++ {
		m := legal.moves[i]
		undo := pos.MakeMove(m)
		s.pushHash(pos.hash)
		reduction := 0
		if depth >= 3 && i >= 4 && !inCheck && m.Captured == 0 && m.Promo == 0 {
			reduction = 1
		}
		var score int
		var complete bool
		if i == 0 {
			score, complete = s.alphaBeta(ctx, pos, depth-1, -beta, -alpha, ply+1, true, deadline)
			score = -score
		} else {
			score, complete = s.alphaBeta(ctx, pos, depth-1-reduction, -alpha-1, -alpha, ply+1, true, deadline)
			score = -score
			if complete && score > alpha {
				score, complete = s.alphaBeta(ctx, pos, depth-1, -beta, -alpha, ply+1, true, deadline)
				score = -score
			}
		}
		s.popHash()
		pos.UndoMove(m, undo)
		if !complete {
			return 0, false
		}
		if score > bestScore {
			bestScore = score
			bestMove = m
		}
		if score > alpha {
			alpha = score
		}
		if alpha >= beta {
			if m.Captured == 0 && !m.EnPassant {
				s.killers[ply][1] = s.killers[ply][0]
				s.killers[ply][0] = m
				s.bumpHistory(pos.sideToMove, m, depth)
			}
			break
		}
	}
	s.storeTT(pos.hash, depth, bestScore, bestMove, boundType(bestScore, origAlpha, beta))
	return bestScore, true
}

func (s *Searcher) quiescence(ctx context.Context, pos *Position, alpha, beta, ply int, deadline time.Time) (int, bool) {
	s.nodes++
	if s.nodes&2047 == 0 && s.shouldStop(ctx, deadline) {
		return 0, false
	}
	standPat := evaluate(*pos)
	if standPat >= beta {
		return standPat, true
	}
	if standPat > alpha {
		alpha = standPat
	}

	var captures MoveList
	pos.generateLegalMovesTo(&captures, true)
	if captures.count == 0 {
		return standPat, true
	}
	s.orderCaptures(&captures)

	for i := 0; i < captures.count; i++ {
		m := captures.moves[i]
		seeMargin := pieceValues[abs(m.Captured)] + 120
		if m.Captured == 0 || standPat+seeMargin < alpha {
			continue
		}
		undo := pos.MakeMove(m)
		s.pushHash(pos.hash)
		score, complete := s.quiescence(ctx, pos, -beta, -alpha, ply+1, deadline)
		s.popHash()
		pos.UndoMove(m, undo)
		if !complete {
			return 0, false
		}
		score = -score
		if score >= beta {
			return score, true
		}
		if score > alpha {
			alpha = score
		}
	}
	return alpha, true
}

func terminalScore(pos Position, depth int) int {
	if pos.InCheck(pos.sideToMove) {
		return -mateScore + (64 - depth)
	}
	return 0
}

func evaluate(pos Position) int {
	score := 0
	for sq, piece := range pos.board {
		if piece == empty {
			continue
		}
		color := colorOf(piece)
		pt := abs(piece)
		v := pieceValues[pt]
		ps := pieceSquare(pt, sq, color)
		score += color * (v + ps)
	}
	if pos.sideToMove == white {
		return score
	}
	return -score
}

func pieceSquare(pieceType, sq, color int) int {
	idx := sq
	if color == black {
		idx = mirrorSq(sq)
	}
	switch pieceType {
	case pawn:
		return pstPawn[idx]
	case knight:
		return pstKnight[idx]
	case bishop:
		return pstBishop[idx]
	case rook:
		return pstRook[idx]
	case queen:
		return pstQueen[idx]
	case king:
		return pstKing[idx]
	default:
		return 0
	}
}

func searchStopped(ctx context.Context, deadline time.Time) bool {
	if deadline.IsZero() {
		select {
		case <-ctx.Done():
			return true
		default:
			return false
		}
	}
	if time.Now().After(deadline) {
		return true
	}
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}

func (s *Searcher) shouldStop(ctx context.Context, deadline time.Time) bool {
	return searchStopped(ctx, deadline)
}

func computeDeadline(side int, limits SearchLimits) time.Time {
	if limits.MoveTime > 0 {
		return time.Now().Add(limits.MoveTime)
	}
	if limits.Infinite {
		return time.Time{}
	}
	remaining := limits.WTime
	if side == black {
		remaining = limits.BTime
	}
	if remaining <= 0 {
		return time.Time{}
	}
	budget := remaining / 30
	if budget < 50*time.Millisecond {
		budget = 50 * time.Millisecond
	}
	if budget > 3*time.Second {
		budget = 3 * time.Second
	}
	return time.Now().Add(budget)
}

func (s *Searcher) pushHash(hash uint64) {
	if s.repCount < len(s.repeat) {
		s.repeat[s.repCount] = hash
		s.repCount++
	}
}

func (s *Searcher) popHash() {
	if s.repCount > 0 {
		s.repCount--
	}
}

func (s *Searcher) isRepetition(hash uint64) bool {
	for i := s.repCount - 2; i >= 0; i -= 2 {
		if s.repeat[i] == hash {
			return true
		}
	}
	return false
}

func ttIndex(hash uint64, size int) int {
	return int(hash) & (size - 1)
}

func (s *Searcher) probeTT(hash uint64, depth, alpha, beta, ply int) (int, bool, Move) {
	entry := s.tt[ttIndex(hash, len(s.tt))]
	if entry.key != hash {
		return 0, false, Move{}
	}
	score := entry.score
	if score > mateScore-128 {
		score -= ply
	} else if score < -mateScore+128 {
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

func (s *Searcher) probeMove(hash uint64) (Move, bool) {
	entry := s.tt[ttIndex(hash, len(s.tt))]
	if entry.key != hash {
		return Move{}, false
	}
	return entry.move, true
}

func (s *Searcher) storeTT(hash uint64, depth, score int, move Move, flag uint8) {
	idx := ttIndex(hash, len(s.tt))
	entry := &s.tt[idx]
	if score > mateScore-128 {
		score += 1
	} else if score < -mateScore+128 {
		score -= 1
	}
	if entry.key != hash || int(entry.depth) <= depth || entry.age != s.age {
		*entry = TTEntry{
			key:   hash,
			move:  move,
			score: score,
			depth: int8(depth),
			flag:  flag,
			age:   s.age,
		}
	}
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

func (s *Searcher) orderMoves(moves *MoveList, ttMove Move, ply int, side int) {
	scores := [256]int{}
	if ply < 0 {
		ply = 0
	}
	sideIdx := 0
	if side == black {
		sideIdx = 1
	}
	for i := 0; i < moves.count; i++ {
		m := moves.moves[i]
		score := 0
		if sameMove(m, ttMove) {
			score += 2_000_000
		}
		if m.Captured != 0 {
			score += 500_000 + 16*pieceValues[abs(m.Captured)]
		} else {
			if sameMove(m, s.killers[ply][0]) {
				score += 90_000
			} else if sameMove(m, s.killers[ply][1]) {
				score += 80_000
			}
			score += s.history[sideIdx][m.From][m.To]
		}
		scores[i] = score
	}
	for i := 0; i < moves.count-1; i++ {
		best := i
		for j := i + 1; j < moves.count; j++ {
			if scores[j] > scores[best] {
				best = j
			}
		}
		if best != i {
			moves.moves[i], moves.moves[best] = moves.moves[best], moves.moves[i]
			scores[i], scores[best] = scores[best], scores[i]
		}
	}
}

func (s *Searcher) orderCaptures(moves *MoveList) {
	scores := [256]int{}
	for i := 0; i < moves.count; i++ {
		m := moves.moves[i]
		attacker := pawn
		if m.Promo != 0 {
			attacker = m.Promo
		}
		scores[i] = 16*pieceValues[abs(m.Captured)] - pieceValues[attacker]
	}
	for i := 0; i < moves.count-1; i++ {
		best := i
		for j := i + 1; j < moves.count; j++ {
			if scores[j] > scores[best] {
				best = j
			}
		}
		if best != i {
			moves.moves[i], moves.moves[best] = moves.moves[best], moves.moves[i]
			scores[i], scores[best] = scores[best], scores[i]
		}
	}
}

func (s *Searcher) bumpHistory(side int, m Move, depth int) {
	sideIdx := 0
	if side == black {
		sideIdx = 1
	}
	cur := s.history[sideIdx][m.From][m.To]
	cur += depth * depth
	if cur > maxHistoryScore {
		cur = maxHistoryScore
	}
	s.history[sideIdx][m.From][m.To] = cur
}

func sameMove(a, b Move) bool {
	return a.From == b.From && a.To == b.To && a.Promo == b.Promo && a.EnPassant == b.EnPassant && a.Castle == b.Castle
}

func hasNonPawnMaterial(pos Position, side int) bool {
	for _, piece := range pos.board {
		if piece == empty || colorOf(piece) != side {
			continue
		}
		pt := abs(piece)
		if pt != pawn && pt != king {
			return true
		}
	}
	return false
}

func (p Position) computeHash() uint64 {
	var h uint64
	for sq, piece := range p.board {
		if piece == empty {
			continue
		}
		h ^= zobristPiece[pieceToHashIndex(piece)][sq]
	}
	h ^= zobristCastling[p.castlingRights]
	h ^= zobristEPFile[epFileIndex(p.epSquare)]
	if p.sideToMove == black {
		h ^= zobristSide
	}
	return h
}

func pieceToHashIndex(piece int) int {
	if piece > 0 {
		return piece - 1
	}
	return 6 + (-piece - 1)
}

func epFileIndex(epSquare int) int {
	if epSquare < 0 || epSquare >= 64 {
		return 0
	}
	return (epSquare % 8) + 1
}

func perft(pos *Position, depth int) int64 {
	if depth == 0 {
		return 1
	}
	var moves MoveList
	pos.generateLegalMovesTo(&moves, false)
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

func parseUCIMove(moveStr string, pos Position) (Move, bool) {
	if len(moveStr) < 4 {
		return Move{}, false
	}
	from, ok1 := squareFromString(moveStr[:2])
	to, ok2 := squareFromString(moveStr[2:4])
	if !ok1 || !ok2 {
		return Move{}, false
	}
	promo := 0
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

	legal := pos.GenerateLegalMoves()
	for _, m := range legal {
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
	if m.Promo != 0 {
		s += string(pieceToPromoRune(m.Promo))
	}
	return s
}

func pieceToPromoRune(piece int) rune {
	switch piece {
	case queen:
		return 'q'
	case rook:
		return 'r'
	case bishop:
		return 'b'
	case knight:
		return 'n'
	default:
		return 'q'
	}
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
	file := sq % 8
	rank := sq / 8
	return string([]byte{byte('a' + file), byte('1' + rank)})
}

func mirrorSq(sq int) int {
	file := sq % 8
	rank := sq / 8
	return (7-rank)*8 + file
}

func inBoard(sq int) bool { return sq >= 0 && sq < 64 }

func colorOf(piece int) int {
	if piece > 0 {
		return white
	}
	if piece < 0 {
		return black
	}
	return 0
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func sameKnightPattern(from, to int) bool {
	df := abs((from % 8) - (to % 8))
	dr := abs((from / 8) - (to / 8))
	return (df == 1 && dr == 2) || (df == 2 && dr == 1)
}

func sameKingPattern(from, to int) bool {
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
	default:
		return false
	}
}
