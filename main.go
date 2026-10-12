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
)

type Move struct {
	From      int
	To        int
	Promo     int
	EnPassant bool
	Castle    bool
}

type Undo struct {
	captured       int
	castlingRights int
	epSquare       int
	halfmove       int
	fullmove       int
}

type Position struct {
	board          [64]int
	sideToMove     int
	castlingRights int
	epSquare       int
	halfmove       int
	fullmove       int
}

type SearchLimits struct {
	Depth    int
	MoveTime time.Duration
	Infinite bool
	WTime    time.Duration
	BTime    time.Duration
}

type Engine struct {
	mu     sync.Mutex
	pos    Position
	cancel context.CancelFunc
	busy   bool
	wg     sync.WaitGroup
}

var (
	knightOffsets = []int{17, 15, 10, 6, -17, -15, -10, -6}
	bishopDirs    = []int{9, 7, -9, -7}
	rookDirs      = []int{8, -8, 1, -1}
	kingOffsets   = []int{8, -8, 1, -1, 9, 7, -9, -7}

	pieceValues = map[int]int{pawn: 100, knight: 320, bishop: 330, rook: 500, queen: 900, king: 0}

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
		fmt.Println("id author Copilot")
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
	limits := SearchLimits{Depth: 6}
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
	undo := Undo{captured: p.board[m.To], castlingRights: p.castlingRights, epSquare: p.epSquare, halfmove: p.halfmove, fullmove: p.fullmove}
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

	p.board[m.From] = empty

	if m.EnPassant {
		if p.sideToMove == white {
			capSq := m.To - 8
			undo.captured = p.board[capSq]
			p.board[capSq] = empty
		} else {
			capSq := m.To + 8
			undo.captured = p.board[capSq]
			p.board[capSq] = empty
		}
	}

	if m.Castle {
		switch m.To {
		case 6:
			p.board[5] = p.board[7]
			p.board[7] = empty
		case 2:
			p.board[3] = p.board[0]
			p.board[0] = empty
		case 62:
			p.board[61] = p.board[63]
			p.board[63] = empty
		case 58:
			p.board[59] = p.board[56]
			p.board[56] = empty
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

	if movingType == pawn && abs(m.To-m.From) == 16 {
		if p.sideToMove == white {
			p.epSquare = m.From + 8
		} else {
			p.epSquare = m.From - 8
		}
	}

	p.updateCastlingRights(m, movingType, undo.captured)
	p.sideToMove = -p.sideToMove
	return undo
}

func (p *Position) UndoMove(m Move, u Undo) {
	p.sideToMove = -p.sideToMove
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
		if p.sideToMove == white {
			capSq := m.To - 8
			p.board[m.To] = empty
			p.board[capSq] = -pawn
		} else {
			capSq := m.To + 8
			p.board[m.To] = empty
			p.board[capSq] = pawn
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

	p.castlingRights = u.castlingRights
	p.epSquare = u.epSquare
	p.halfmove = u.halfmove
	p.fullmove = u.fullmove
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
	pseudo := p.generatePseudoMoves()
	moves := make([]Move, 0, len(pseudo))
	for _, m := range pseudo {
		cp := p
		u := cp.MakeMove(m)
		if !cp.InCheck(-cp.sideToMove) {
			moves = append(moves, m)
		}
		cp.UndoMove(m, u)
	}
	return moves
}

func (p Position) generatePseudoMoves() []Move {
	moves := make([]Move, 0, 64)
	for sq, piece := range p.board {
		if piece == empty || colorOf(piece) != p.sideToMove {
			continue
		}
		t := abs(piece)
		switch t {
		case pawn:
			p.generatePawnMoves(sq, &moves)
		case knight:
			p.generateKnightMoves(sq, &moves)
		case bishop:
			p.generateSliderMoves(sq, bishopDirs, &moves)
		case rook:
			p.generateSliderMoves(sq, rookDirs, &moves)
		case queen:
			p.generateSliderMoves(sq, bishopDirs, &moves)
			p.generateSliderMoves(sq, rookDirs, &moves)
		case king:
			p.generateKingMoves(sq, &moves)
		}
	}
	return moves
}

func (p Position) generatePawnMoves(sq int, moves *[]Move) {
	rank := sq / 8
	file := sq % 8
	dir := 8
	startRank := 1
	promoRank := 6
	enemy := black
	if p.sideToMove == black {
		dir = -8
		startRank = 6
		promoRank = 1
		enemy = white
	}

	one := sq + dir
	if inBoard(one) && p.board[one] == empty {
		if rank == promoRank {
			for _, promo := range []int{queen, rook, bishop, knight} {
				*moves = append(*moves, Move{From: sq, To: one, Promo: promo})
			}
		} else {
			*moves = append(*moves, Move{From: sq, To: one})
			if rank == startRank {
				two := sq + 2*dir
				if inBoard(two) && p.board[two] == empty {
					*moves = append(*moves, Move{From: sq, To: two})
				}
			}
		}
	}

	for _, df := range []int{-1, 1} {
		nf := file + df
		if nf < 0 || nf > 7 {
			continue
		}
		to := sq + dir + df
		if !inBoard(to) {
			continue
		}
		if p.board[to] != empty && colorOf(p.board[to]) == enemy {
			if rank == promoRank {
				for _, promo := range []int{queen, rook, bishop, knight} {
					*moves = append(*moves, Move{From: sq, To: to, Promo: promo})
				}
			} else {
				*moves = append(*moves, Move{From: sq, To: to})
			}
		}
		if to == p.epSquare {
			*moves = append(*moves, Move{From: sq, To: to, EnPassant: true})
		}
	}
}

func (p Position) generateKnightMoves(sq int, moves *[]Move) {
	for _, off := range knightOffsets {
		to := sq + off
		if !inBoard(to) || !sameKnightPattern(sq, to) {
			continue
		}
		if p.board[to] == empty || colorOf(p.board[to]) != p.sideToMove {
			*moves = append(*moves, Move{From: sq, To: to})
		}
	}
}

func (p Position) generateSliderMoves(sq int, dirs []int, moves *[]Move) {
	for _, d := range dirs {
		to := sq
		for {
			next := to + d
			if !inBoard(next) || !sameLine(to, next, d) {
				break
			}
			to = next
			if p.board[to] == empty {
				*moves = append(*moves, Move{From: sq, To: to})
				continue
			}
			if colorOf(p.board[to]) != p.sideToMove {
				*moves = append(*moves, Move{From: sq, To: to})
			}
			break
		}
	}
}

func (p Position) generateKingMoves(sq int, moves *[]Move) {
	for _, off := range kingOffsets {
		to := sq + off
		if !inBoard(to) || !sameKingPattern(sq, to) {
			continue
		}
		if p.board[to] == empty || colorOf(p.board[to]) != p.sideToMove {
			*moves = append(*moves, Move{From: sq, To: to})
		}
	}

	if p.sideToMove == white && sq == 4 {
		if p.castlingRights&castleWK != 0 && p.board[5] == empty && p.board[6] == empty && !p.InCheck(white) && !p.squareAttacked(5, black) && !p.squareAttacked(6, black) {
			*moves = append(*moves, Move{From: 4, To: 6, Castle: true})
		}
		if p.castlingRights&castleWQ != 0 && p.board[3] == empty && p.board[2] == empty && p.board[1] == empty && !p.InCheck(white) && !p.squareAttacked(3, black) && !p.squareAttacked(2, black) {
			*moves = append(*moves, Move{From: 4, To: 2, Castle: true})
		}
	}
	if p.sideToMove == black && sq == 60 {
		if p.castlingRights&castleBK != 0 && p.board[61] == empty && p.board[62] == empty && !p.InCheck(black) && !p.squareAttacked(61, white) && !p.squareAttacked(62, white) {
			*moves = append(*moves, Move{From: 60, To: 62, Castle: true})
		}
		if p.castlingRights&castleBQ != 0 && p.board[59] == empty && p.board[58] == empty && p.board[57] == empty && !p.InCheck(black) && !p.squareAttacked(59, white) && !p.squareAttacked(58, white) {
			*moves = append(*moves, Move{From: 60, To: 58, Castle: true})
		}
	}
}

func (p Position) InCheck(color int) bool {
	kingSq := -1
	for sq, piece := range p.board {
		if piece == color*king {
			kingSq = sq
			break
		}
	}
	if kingSq == -1 {
		return false
	}
	return p.squareAttacked(kingSq, -color)
}

func (p Position) squareAttacked(sq int, byColor int) bool {
	file := sq % 8

	if byColor == white {
		for _, off := range []int{-7, -9} {
			from := sq + off
			if inBoard(from) && abs((from%8)-file) == 1 && p.board[from] == pawn {
				return true
			}
		}
	} else {
		for _, off := range []int{7, 9} {
			from := sq + off
			if inBoard(from) && abs((from%8)-file) == 1 && p.board[from] == -pawn {
				return true
			}
		}
	}

	for _, off := range knightOffsets {
		from := sq + off
		if inBoard(from) && sameKnightPattern(sq, from) && p.board[from] == byColor*knight {
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

	for _, off := range kingOffsets {
		from := sq + off
		if inBoard(from) && sameKingPattern(sq, from) && p.board[from] == byColor*king {
			return true
		}
	}

	return false
}

func SearchBestMove(ctx context.Context, pos Position, limits SearchLimits) Move {
	moves := pos.GenerateLegalMoves()
	if len(moves) == 0 {
		return Move{From: -1, To: -1}
	}

	maxDepth := limits.Depth
	if maxDepth <= 0 {
		maxDepth = 6
	}
	deadline := time.Time{}
	if limits.MoveTime > 0 {
		deadline = time.Now().Add(limits.MoveTime)
	} else if !limits.Infinite {
		remaining := limits.WTime
		if pos.sideToMove == black {
			remaining = limits.BTime
		}
		if remaining > 0 {
			budget := remaining / 30
			if budget < 50*time.Millisecond {
				budget = 50 * time.Millisecond
			}
			if budget > 2*time.Second {
				budget = 2 * time.Second
			}
			deadline = time.Now().Add(budget)
		}
	}

	bestMove := moves[0]
	bestScore := math.MinInt

	for depth := 1; depth <= maxDepth; depth++ {
		if searchStopped(ctx, deadline) {
			break
		}
		score, move, complete := searchRoot(ctx, pos, depth, deadline)
		if !complete {
			break
		}
		bestScore = score
		bestMove = move
		fmt.Printf("info depth %d score cp %d pv %s\n", depth, bestScore, bestMove.UCI())
	}

	return bestMove
}

func searchRoot(ctx context.Context, pos Position, depth int, deadline time.Time) (int, Move, bool) {
	moves := pos.GenerateLegalMoves()
	if len(moves) == 0 {
		return terminalScore(pos, depth), Move{From: -1, To: -1}, true
	}

	best := moves[0]
	bestScore := math.MinInt
	alpha, beta := -mateScore, mateScore

	for _, m := range moves {
		if searchStopped(ctx, deadline) {
			return bestScore, best, false
		}
		undo := pos.MakeMove(m)
		score, complete := alphaBeta(ctx, &pos, depth-1, -beta, -alpha, deadline)
		pos.UndoMove(m, undo)
		if !complete {
			return bestScore, best, false
		}
		score = -score
		if score > bestScore {
			bestScore = score
			best = m
		}
		if score > alpha {
			alpha = score
		}
	}

	return bestScore, best, true
}

func alphaBeta(ctx context.Context, pos *Position, depth int, alpha, beta int, deadline time.Time) (int, bool) {
	if searchStopped(ctx, deadline) {
		return 0, false
	}
	if depth == 0 {
		return evaluate(*pos), true
	}

	moves := pos.GenerateLegalMoves()
	if len(moves) == 0 {
		return terminalScore(*pos, depth), true
	}

	best := math.MinInt
	for _, m := range moves {
		undo := pos.MakeMove(m)
		score, complete := alphaBeta(ctx, pos, depth-1, -beta, -alpha, deadline)
		pos.UndoMove(m, undo)
		if !complete {
			return 0, false
		}
		score = -score
		if score > best {
			best = score
		}
		if best > alpha {
			alpha = best
		}
		if alpha >= beta {
			break
		}
	}
	return best, true
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
