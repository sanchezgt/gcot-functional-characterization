---
id: E055
plantilla: P10
dominio: informatica
estilo: terso
respuesta: 625
enunciado: A sensor produces readings that can take 1,000 distinct values. Each reading is encoded with the minimum number of bits needed to distinguish all values, and the bits of consecutive readings are packed with no padding. How many bytes are needed to store 500 readings?
---
## PROGRESSION:ADVANCES
Need bytes for 500 packed readings, each with the minimum bit width for 1,000 distinct values.

## PROGRESSION:ADVANCES
Bits per reading: smallest b with 2^b >= 1,000. 2^9 = 512 < 1,000; 2^10 = 1,024 >= 1,000. So b = 10 bits.

## VERIFICATION:CONFIRMED | explicita
Check by logarithm: log2(1,000) is about 9.97, and the bit count must be a whole number at least that large, so 10. Confirmed, 10 bits.

## PROGRESSION:ADVANCES
Total bits for 500 readings: 500 x 10 = 5,000 bits.

## ERROR
Bytes: 5,000 bits is 500 bytes.

## VERIFICATION:REFUTED | explicita
Check the conversion: a byte is 8 bits, not 10; the step above divided by the bits per reading instead of the bits per byte. Wrong.

## CORRECTION:RECOVERED | explicita
Correcting it: 5,000 / 8 = 625 bytes.

## SUPPORT
Packing without padding means the bits of one reading continue directly into the next byte, so only the total bit count matters.

## PROGRESSION:REACHES_OUTPUT
Storage: 625 bytes. Answer: 625.
