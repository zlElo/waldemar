package main

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

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
