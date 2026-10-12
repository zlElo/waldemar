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

	func TestPerftStartPosition(t *testing.T) {
		pos, err := NewPositionFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
		if err != nil {
			t.Fatalf("fen parse failed: %v", err)
		}
		tests := []struct {
			depth int
			want  int64
		}{
			{1, 20},
			{2, 400},
			{3, 8902},
		}
		for _, tt := range tests {
			cp := pos.Clone()
			got := perft(&cp, tt.depth)
			if got != tt.want {
				t.Fatalf("perft depth %d mismatch: got %d want %d", tt.depth, got, tt.want)
			}
		}
	}

	func TestSearchStableAtFixedDepth(t *testing.T) {
		pos, err := NewPositionFromFEN("r1bqkbnr/pppp1ppp/2n5/4p3/1bB1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4")
		if err != nil {
			t.Fatalf("fen parse failed: %v", err)
		}
		first := SearchBestMove(context.Background(), pos, SearchLimits{Depth: 5, MoveTime: 0})
		for i := 0; i < 5; i++ {
			got := SearchBestMove(context.Background(), pos, SearchLimits{Depth: 5, MoveTime: 0})
			if got.UCI() != first.UCI() {
				t.Fatalf("non-deterministic result at depth 5: got %s want %s", got.UCI(), first.UCI())
			}
		}
	}

	func BenchmarkMoveGenerationStartPos(b *testing.B) {
		pos, err := NewPositionFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
		if err != nil {
			b.Fatalf("fen parse failed: %v", err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = pos.GenerateLegalMoves()
		}
	}

	func BenchmarkSearchDepth5StartPos(b *testing.B) {
		pos, err := NewPositionFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
		if err != nil {
			b.Fatalf("fen parse failed: %v", err)
		}
		limits := SearchLimits{Depth: 5, MoveTime: 0}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = SearchBestMove(context.Background(), pos, limits)
		}
	}
	m := SearchBestMove(context.Background(), pos, SearchLimits{Depth: 2, MoveTime: 100 * time.Millisecond})
	if m.From < 0 || m.To < 0 {
		t.Fatalf("expected a move, got %s", m.UCI())
	}
	if _, ok := parseUCIMove(m.UCI(), pos); !ok {
		t.Fatalf("search returned illegal move %s", m.UCI())
	}
}
