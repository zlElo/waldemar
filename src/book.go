package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

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
