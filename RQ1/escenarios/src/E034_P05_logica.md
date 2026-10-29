---
id: E034
plantilla: P05
dominio: logica
estilo: terso
respuesta: A
enunciado: Five people, A, B, C, D, and E, stand in a single line. E is first in line and C is last. A stands immediately in front of B, and D stands immediately behind B. Who is second in line?
---
## PROGRESSION:ADVANCES
Need the person in position 2. Five positions. Constraints: E first, C last, A immediately before B, D immediately after B.

## ALTERNATIVE:REJECTED | explicita
Alternatively, one could list all 120 orderings of five people and strike out those that violate a constraint. But two of the constraints fix positions outright, so placing those people first will pin the line down in a few steps. The brute-force listing is not worth it.

## PROGRESSION:ADVANCES
E is in position 1 and C is in position 5. Positions 2, 3, and 4 remain for A, B, and D.

## PROGRESSION:ADVANCES
A is immediately before B and D is immediately after B, so A, B, D form a consecutive block in that order. The only room for a block of three is positions 2, 3, 4. So A is second, B third, D fourth.

## PROGRESSION:REACHES_OUTPUT
Line: E, A, B, D, C. Second in line: A. Answer: A.
