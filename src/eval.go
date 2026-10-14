package main

import (
	"math/bits"
)

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
