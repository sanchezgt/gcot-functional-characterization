---
id: E064
plantilla: P12
dominio: informatica
estilo: narrativo
respuesta: 0
enunciado: A communication system appends an even-parity bit to each 7-bit data word, so that the total number of 1s in the transmitted 8-bit word is even; the receiver uses this bit to check each word. What parity bit is appended to the data word 1011001?
---
## PROGRESSION:ADVANCES | distractor
First, I recall what the parity bit is for: the sender appends one bit so that the total number of 1s in the 8-bit word is even, and the receiver can then check each word by counting its 1s.

## PROGRESSION:ADVANCES
I count the 1s in 1011001: the first, third, fourth, and seventh bits are 1, so there are 4 ones.

## PROGRESSION:ADVANCES
Four is already even, so the parity bit must be 0 to keep the total even.

## SUPPORT | distractor
This is why a parity check catches any single flipped bit: one flip changes the count of 1s from even to odd, the receiver's verification step sees an odd count and flags the word, and the sender is asked to retransmit it while the next words wait in the queue.

## PROGRESSION:REACHES_OUTPUT
The appended parity bit is 0, and the transmitted word is 10110010.

Final answer: 0.
