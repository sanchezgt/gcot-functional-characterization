---
id: E104
plantilla: P12
dominio: informatica
estilo: deepseek
respuesta: 31
enunciado: A client sends a request to a server and uses exponential backoff: after each failed attempt it waits before retrying, and the wait doubles each time, starting at 1 second after the first failure. Each response is checked with a checksum to decide whether the attempt failed. The request fails 5 times in a row and succeeds on the 6th attempt. Ignoring the time spent on the attempts themselves, how many seconds does the client spend waiting in total?
---
## PROGRESSION:ADVANCES | distractor
We are given: "A client uses exponential backoff: after each failed attempt it waits before retrying, and the wait doubles each time, starting at 1 second. The request fails 5 times and succeeds on the 6th attempt. How many seconds does the client spend waiting in total?"

We need the total wait. There is one wait after each failure, so 5 waits come before the 6th attempt.

## PROGRESSION:ADVANCES
The waits double each time: 1, 2, 4, 8, and 16 seconds.

## SUPPORT | distractor
The checksum check on each response only decides whether that attempt counts as a failure; it adds no wait of its own, and the problem ignores the time of the attempts, so the total is just the sum of the backoff waits.

## PROGRESSION:ADVANCES
Total: 1 + 2 + 4 + 8 + 16 = 31 seconds.

## PROGRESSION:REACHES_OUTPUT
So the client spends 31 seconds waiting before the 6th attempt goes through.

Thus final answer: 31.
