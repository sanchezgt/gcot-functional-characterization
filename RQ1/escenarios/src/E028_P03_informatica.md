---
id: E028
plantilla: P03
dominio: informatica
estilo: deepseek
respuesta: 3
enunciado: A sorted array contains 7 distinct integers. In the worst case, how many comparisons does binary search make to determine whether a given value is in the array? Assume one comparison per probe.
---
## PROGRESSION:ADVANCES
We are given: "A sorted array contains 7 distinct integers. In the worst case, how many comparisons does binary search make to determine whether a given value is in the array?"

We need the worst-case number of probes of binary search for n = 7.

## ERROR
Binary search looks at half the array in the worst case, so the number of comparisons is n / 2 = 7 / 2 = 3.5, which rounds up to 4 comparisons.

## VERIFICATION:REFUTED | explicita
Wait, let me check that by simulating the search. Probe 1 looks at the middle element and leaves 3 candidates. Probe 2 looks at the middle of those and leaves 1 candidate. Probe 3 looks at that last candidate and decides. That is 3 comparisons, not 4; binary search does not inspect half of the elements, it discards half of them at every step.

## CORRECTION:RECOVERED | explicita
Correcting the formula: each probe halves the remaining candidates, so the worst case is the number of probes needed to exhaust them, ceil(log2(n + 1)) = ceil(log2(8)) = 3 comparisons.

## SUPPORT
Each probe removes the middle element together with one half of the remaining candidates, which is what makes the search logarithmic.

## PROGRESSION:REACHES_OUTPUT
Thus final answer: 3.
