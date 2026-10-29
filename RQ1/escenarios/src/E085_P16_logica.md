---
id: E085
plantilla: P16
dominio: logica
estilo: escolar
respuesta: 45
enunciado: At a meeting, each of 10 people shakes hands exactly once with every other person. How many handshakes take place?
---
## PROGRESSION:ADVANCES
We need the number of handshakes among 10 people when every pair shakes hands exactly once.

## PROGRESSION:ABANDONED
Count person by person: the first person shakes 9 hands, the second person has 8 handshakes not yet counted, the third has 7, and so on, so the total is 9 + 8 + 7 + ...

## ALTERNATIVE:ADOPTED | implicita
Each handshake corresponds to exactly one pair of people, so the total is the number of ways to choose 2 people out of 10, C(10, 2) = 10 x 9 / 2.

## PROGRESSION:ADVANCES
C(10, 2) = 90 / 2 = 45.

## SUPPORT
The division by 2 appears because choosing person A and then person B gives the same pair, and the same handshake, as choosing B and then A.

## PROGRESSION:REACHES_OUTPUT
There are 45 handshakes.

Answer: 45.
