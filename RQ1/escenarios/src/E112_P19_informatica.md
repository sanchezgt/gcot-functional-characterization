---
id: E112
plantilla: P19
dominio: informatica
estilo: narrativo
respuesta: -10
enunciado: What decimal value does the 8-bit pattern 11110110 represent when it is read as a two's complement signed integer?
---
## PROGRESSION:ADVANCES
First, I note what is asked: the decimal value of the 8-bit pattern 11110110 read as a two's complement number. Its leading bit is 1, so the number is negative.

## ALTERNATIVE:REJECTED | implicita
One route would invert all eight bits and add one to obtain the magnitude, then attach the minus sign. Giving the leading bit a weight of -128 instead yields the value in a single sum, without rewriting the pattern twice, so I leave the inversion route aside.

## PROGRESSION:ADVANCES
Next, I assign the weights from left to right: -128, 64, 32, 16, 8, 4, 2, 1. The bits that are set carry the weights -128, 64, 32, 16, 4, and 2.

## PROGRESSION:ADVANCES
Adding the positive weights gives 64 + 32 + 16 + 4 + 2 = 118, and with the leading bit the value is -128 + 118 = -10.

## VERIFICATION:CONFIRMED | implicita
Read as an unsigned number, the same pattern is 128 + 118 = 246, and an 8-bit pattern whose leading bit is 1 stands, in two's complement, for its unsigned value minus 256: 246 - 256 = -10, the same value as the weighted sum.

## PROGRESSION:REACHES_OUTPUT
The pattern 11110110 represents -10 in 8-bit two's complement.

Final answer: -10.
