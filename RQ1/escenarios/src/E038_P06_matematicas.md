---
id: E038
plantilla: P06
dominio: matematicas
estilo: narrativo
respuesta: 2550
enunciado: What is the sum of the first 50 positive even numbers?
---
## PROGRESSION:ADVANCES
First, I pin down which numbers are meant: the first 50 positive even numbers are 2, 4, 6, and so on up to 100.

## ALTERNATIVE:REJECTED | implicita
Adding the fifty numbers one after another would reach the total, but each even number is twice an integer from 1 to 50, so the whole sum collapses into one closed formula and the fifty additions are unnecessary.

## PROGRESSION:ADVANCES
Writing each term as 2k, the sum is 2(1 + 2 + ... + 50).

## PROGRESSION:ADVANCES
The sum of the integers from 1 to n is n(n + 1)/2, so 1 + 2 + ... + 50 = 50 x 51 / 2 = 1,275.

## PROGRESSION:ADVANCES
Doubling it: 2 x 1,275 = 2,550.

## SUPPORT
The formula n(n + 1)/2 comes from pairing the first term with the last, the second with the second to last, and so on: there are n/2 pairs and each sums to n + 1.

## PROGRESSION:REACHES_OUTPUT
The sum of the first 50 positive even numbers is 2,550.

Final answer: 2550.
