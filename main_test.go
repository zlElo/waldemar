package main

import (
	"context"
	"testing"
	"time"
)

func TestStartPosMoveCount(t *testing.T) {
	pos, err := NewPositionFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	if err != nil {
		t.Fatalf("fen parse failed: %v", err)
	}
	moves := pos.GenerateLegalMoves()
	if len(moves) != 20 {
		t.Fatalf("expected 20 legal moves, got %d", len(moves))
	}
}

func TestCastlingGenerated(t *testing.T) {
	pos, err := NewPositionFromFEN("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
	if err != nil {
		t.Fatalf("fen parse failed: %v", err)
	}
	moves := pos.GenerateLegalMoves()
	want := map[string]bool{"e1g1": false, "e1c1": false}
	for _, m := range moves {
		if _, ok := want[m.UCI()]; ok {
			want[m.UCI()] = true
		}
	}
	for k, v := range want {
		if !v {
			t.Fatalf("expected castling move %s", k)
		}
	}
}

func TestEnPassantGenerated(t *testing.T) {
	pos, err := NewPositionFromFEN("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
	if err != nil {
		t.Fatalf("fen parse failed: %v", err)
	}
	moves := pos.GenerateLegalMoves()
	found := false
	for _, m := range moves {
		if m.UCI() == "e5d6" {
			found = true
			if !m.EnPassant {
				t.Fatalf("e5d6 should be marked en passant")
			}
		}
	}
	if !found {
		t.Fatalf("expected en passant move e5d6")
	}
}

func TestSearchReturnsLegalMove(t *testing.T) {
	pos, err := NewPositionFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	if err != nil {
		t.Fatalf("fen parse failed: %v", err)
	}
	m := SearchBestMove(context.Background(), pos, SearchLimits{Depth: 2, MoveTime: 100 * time.Millisecond})
	if m.From < 0 || m.To < 0 {
		t.Fatalf("expected a move, got %s", m.UCI())
	}
	if _, ok := parseUCIMove(m.UCI(), pos); !ok {
		t.Fatalf("search returned illegal move %s", m.UCI())
	}
}
