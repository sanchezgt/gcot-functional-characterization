---
id: E043
plantilla: P07
dominio: logica
estilo: terso
respuesta: 42
enunciado: What is the next number in the sequence 2, 6, 12, 20, 30, ...?
---
## PROGRESSION:ADVANCES
Need the next term of 2, 6, 12, 20, 30.

## PROGRESSION:ABANDONED
Try ratios between consecutive terms: 6/2 = 3, 12/6 = 2, 20/12 = 1.67, 30/20 = 1.5.

## ALTERNATIVE:ADOPTED | explicita
Alternatively, look at differences instead of ratios, since the ratios shrink without a clean rule and continuing would mean guessing the next one: 6 - 2 = 4, 12 - 6 = 6, 20 - 12 = 8, 30 - 20 = 10. The differences go up by 2 each time, which is a clean rule. Switching to the differences and dropping the ratio approach.

## PROGRESSION:ADVANCES
Next difference: 12. Next term: 30 + 12 = 42.

## SUPPORT
Differences that grow by a constant amount mean the terms follow a quadratic rule, so each new difference is the previous one plus 2.

## PROGRESSION:REACHES_OUTPUT
Next number: 42. Answer: 42.
