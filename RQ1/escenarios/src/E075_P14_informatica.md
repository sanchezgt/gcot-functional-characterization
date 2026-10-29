---
id: E075
plantilla: P14
dominio: informatica
estilo: escolar
respuesta: D
enunciado: Consider the following code: for i from 1 to n, for j from 1 to i, perform one constant-time operation. What is the time complexity of this code in terms of n? Options: A. O(log n) B. O(n) C. O(n log n) D. O(n^2)
---
## PROGRESSION:ADVANCES
We need the time complexity of two nested loops: the outer loop runs i from 1 to n, and for each i the inner loop runs j from 1 to i, doing one constant-time operation per iteration.

## ALTERNATIVE:REJECTED | implicita
Timing the code on a few input sizes could suggest a growth rate, but counting the iterations exactly gives a result that holds for every n rather than an estimate.

## ERROR | id=e1
The inner loop runs i times on the i-th pass, which averages to about n/2, so the total work is about n/2 operations, which is O(n).

## VERIFICATION:REFUTED | implicita
About n/2 is the average length of a single inner loop, but there are n outer passes, each running its own inner loop, so n/2 is the work of one pass, not of the whole loop.

## CORRECTION:RECOVERED | implicita | id=c1
The total number of operations is 1 + 2 + ... + n = n(n + 1)/2, which grows as n^2, so the complexity is O(n^2).

## VERIFICATION:CONFIRMED | implicita | objetivo=c1
For n = 4, the passes run 1, 2, 3, and 4 times, 10 operations in all, and 4 x 5 / 2 = 10.

## PROGRESSION:REACHES_OUTPUT
The time complexity is O(n^2), option D.

Answer: D.
