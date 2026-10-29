---
id: E082
plantilla: P16
dominio: informatica
estilo: terso
respuesta: yes
enunciado: Is 97 a prime number? Answer yes or no.
---
## PROGRESSION:ADVANCES
Need to decide whether 97 is prime.

## PROGRESSION:ABANDONED
Trial division by every integer from 2 up to 96: 97 / 2 not an integer, 97 / 3 no, 97 / 4 no...

## ALTERNATIVE:ADOPTED | implicita
Any factorization 97 = a x b has a factor no larger than sqrt(97), just under 10, and every composite factor has a smaller prime factor, so only the primes 2, 3, 5, and 7 need testing.

## PROGRESSION:ADVANCES
97 is odd, so not divisible by 2, and its digit sum is 16, so not divisible by 3. 97 = 5 x 19 + 2 and 97 = 7 x 13 + 6, so neither 5 nor 7 divides 97.

## PROGRESSION:REACHES_OUTPUT
97 is prime. Answer: yes.
