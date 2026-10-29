---
id: E128
plantilla: P13
dominio: informatica
estilo: terso
respuesta: 31
enunciado: A perfect binary tree has every internal node with exactly two children and all leaves at the same depth. If its height is 4, with height measured as the number of edges on the path from the root to a leaf, how many nodes does the tree have?
---
## PROGRESSION:ADVANCES
Need the node count of a perfect binary tree of height 4, height counted in edges.

## ALTERNATIVE:REJECTED | explicita
Alternatively: list the levels and add their sizes one by one. Workable, but the closed formula for a perfect binary tree gives the count in one step. Not using the level-by-level sum.

## ERROR | id=e1
Formula: 2^h - 1 = 2^4 - 1 = 15 nodes.

## VERIFICATION:REFUTED | explicita
Check: with height counted in edges, a height of 4 means levels 0 through 4, five levels in all. 2^h - 1 counts only h levels, so it leaves out the deepest one. Wrong.

## CORRECTION:RECOVERED | explicita | id=c1
Correcting it with h + 1 levels: 2^(h+1) - 1 = 2^5 - 1 = 31 nodes.

## VERIFICATION:CONFIRMED | explicita | objetivo=c1
Check by levels: 1 + 2 + 4 + 8 + 16 = 31. Confirmed.

## PROGRESSION:REACHES_OUTPUT
Node count: 31. Answer: 31.
