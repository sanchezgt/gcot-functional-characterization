---
id: E067
plantilla: P13
dominio: matematicas
estilo: deepseek
respuesta: 6
enunciado: A rectangular garden has an area of 48 square meters, and its length is 2 meters more than its width. What is the width of the garden, in meters?
---
## PROGRESSION:ADVANCES
We are given: "A rectangular garden has an area of 48 square meters, and its length is 2 meters more than its width. What is the width?" Let the width be w. Then the length is w + 2.

## PROGRESSION:ADVANCES
Area = length x width: w(w + 2) = 48, so w^2 + 2w - 48 = 0.

## ALTERNATIVE:REJECTED | explicita
Alternatively, I could solve w^2 + 2w - 48 = 0 with the quadratic formula. But the constant term 48 has an obvious pair of factors that differ by 2, so factoring should be quicker here; I set the formula aside.

## ERROR | id=e1
Factoring: (w - 8)(w + 6) = 0, so w = 8 or w = -6, and since a width is positive, the width is 8 meters.

## VERIFICATION:REFUTED | explicita
Wait, let me check that factorization by expanding it: (w - 8)(w + 6) = w^2 - 2w - 48, but the equation has +2w. So the signs in the factors are reversed.

## CORRECTION:RECOVERED | explicita | id=c1
Correcting it: (w + 8)(w - 6) = w^2 + 2w - 48, so w = -8 or w = 6, and the positive root gives a width of 6 meters.

## VERIFICATION:CONFIRMED | explicita | objetivo=c1
Let me verify against the original conditions: width 6, length 6 + 2 = 8, and area 6 x 8 = 48. Confirmed.

## PROGRESSION:REACHES_OUTPUT
Thus final answer: 6.
