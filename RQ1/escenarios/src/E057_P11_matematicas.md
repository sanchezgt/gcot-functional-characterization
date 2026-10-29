---
id: E057
plantilla: P11
dominio: matematicas
estilo: deepseek
respuesta: 24
enunciado: How many positive divisors does the number 360 have?
---
## PROGRESSION:ADVANCES
We are given: "How many positive divisors does the number 360 have?"

We need to count all positive integers that divide 360.

## ALTERNATIVE:REJECTED | explicita
Alternatively, I could test every integer from 1 to 360 for divisibility and count the hits. That is 360 checks, and it tells me nothing about why the count is what it is. The prime factorization gives the count directly, so I will not do the listing.

## PROGRESSION:ADVANCES | id=f1
Factor 360: 360 = 2 * 180 = 2 * 2 * 90 = 2 * 2 * 2 * 45 = 2^3 * 45, and 45 = 3^2 * 5. So 360 = 2^3 * 3^2 * 5^1.

## PROGRESSION:ADVANCES
The number of divisors is the product of each exponent plus one: (3 + 1)(2 + 1)(1 + 1) = 4 * 3 * 2 = 24.

## SUPPORT
Each divisor is built by choosing a power of 2 from 0 to 3, a power of 3 from 0 to 2, and a power of 5 from 0 to 1, and every combination gives a different divisor.

## VERIFICATION:CONFIRMED | explicita | objetivo=f1
Let me double-check the factorization, since the count depends on it: 2^3 * 3^2 * 5 = 8 * 9 * 5 = 72 * 5 = 360. Confirmed, so the count of 24 stands.

## PROGRESSION:REACHES_OUTPUT
Thus final answer: 24.
