---
id: E111
plantilla: P16
dominio: economia
estilo: terso
respuesta: 75
enunciado: A preferred share pays a fixed dividend of $6 at the end of every year, forever. If investors require a return of 8% per year, what is the share's price today, in dollars?
---
## PROGRESSION:ADVANCES
Need today's price of a share paying $6 a year forever, required return 8%.

## PROGRESSION:ABANDONED
Discount each dividend to today: 6 / 1.08 = 5.56, 6 / 1.08^2 = 5.14, 6 / 1.08^3 = 4.76, ...

## ALTERNATIVE:ADOPTED | implicita
A constant payment received forever is a perpetuity, and its present value is the payment divided by the rate, P = D / r, with no term-by-term sum.

## PROGRESSION:ADVANCES
P = 6 / 0.08 = 75 dollars.

## SUPPORT
P = D / r is the sum of the geometric series of discounted dividends: first term 6 / 1.08, ratio 1 / 1.08, so (6 / 1.08) / (0.08 / 1.08) = 6 / 0.08.

## PROGRESSION:REACHES_OUTPUT
Price: $75. Answer: 75.
