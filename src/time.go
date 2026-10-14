package main

import "time"

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
