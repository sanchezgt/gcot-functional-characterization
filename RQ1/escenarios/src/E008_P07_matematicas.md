---
id: E008
plantilla: P07
dominio: matematicas
estilo: narrativo
respuesta: 15
enunciado: How many three-digit positive integers have digits that add up to 5?
---
## PROGRESSION:ADVANCES
First, I set up what is being counted: three-digit numbers, so the hundreds digit goes from 1 to 9 and the other two digits from 0 to 9, with the three digits summing to 5.

## PROGRESSION:ABANDONED
My first plan is to go through the hundreds digit case by case. If the hundreds digit is 1, the last two digits must sum to 4: 04, 13, 22, 31, 40, which is 5 numbers. If the hundreds digit is 2, the last two digits must sum to 3: 03, 12, 21, 30, which is 4 numbers.

## ALTERNATIVE:ADOPTED | explicita
Alternatively, there is a cleaner route: count the solutions of a + b + c = 5 with a at least 1 and b, c at least 0 using stars and bars. That handles all the cases at once instead of listing them, so I will switch to this approach and leave the case-by-case listing unfinished.

## PROGRESSION:ADVANCES
Substituting a' = a - 1, the equation becomes a' + b + c = 4 with all three variables nonnegative. The number of nonnegative integer solutions is C(4 + 2, 2) = C(6, 2) = 15.

## SUPPORT
Stars and bars applies directly because the upper limit of 9 on each digit can never be reached when the digits add up to only 5.

## PROGRESSION:REACHES_OUTPUT
Therefore there are 15 such three-digit numbers.

Final answer: 15.
