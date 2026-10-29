---
id: E095
plantilla: P18
dominio: informatica
estilo: terso
respuesta: 6 iterations; last value 10
enunciado: How many times does the body of the loop `for (int i = 0; i <= 10; i += 2) { print(i); }` execute, and what is the last value printed?
---
## PROGRESSION:ADVANCES
Need the iteration count and the last printed value: i starts at 0, the loop runs while i <= 10, step 2.

## ERROR | id=e1
Iterations: range 10, step 2, so 10 / 2 = 5.

## PROGRESSION:ADVANCES
Five iterations starting at 0: printed values 0, 2, 4, 6, 8.

## PROGRESSION:ADVANCES
Last printed value: 8. After it, i becomes 10.

## VERIFICATION:REFUTED | explicita | objetivo=e1
Check the count against the loop condition: i <= 10 is true at i = 10, so the body also runs at 10. The values 0, 2, 4, 6, 8, 10 are 6, not 5. The count formula missed the starting value, so 5 is wrong.

## CORRECTION:RECOVERED | explicita | objetivo=e1
Corrected: the count is 10 / 2 + 1 = 6 iterations, printing 0, 2, 4, 6, 8, 10. Last value printed: 10.

## PROGRESSION:REACHES_OUTPUT
Answer: 6 iterations, last value 10.
