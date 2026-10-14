package main

import (
	"math/bits"
)

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
