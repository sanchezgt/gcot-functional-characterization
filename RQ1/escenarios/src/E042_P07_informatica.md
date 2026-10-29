---
id: E042
plantilla: P07
dominio: informatica
estilo: deepseek
respuesta: 6
enunciado: Given the array [-2, 1, -3, 4, -1, 2, 1, -5, 4], what is the largest possible sum of a contiguous (non-empty) subarray?
---
## PROGRESSION:ADVANCES
We are given the array [-2, 1, -3, 4, -1, 2, 1, -5, 4] and need the maximum sum over all contiguous non-empty subarrays.

## PROGRESSION:ABANDONED
One approach is to enumerate subarrays by their starting index and keep the best sum seen. Starting at index 0: -2, then -2 + 1 = -1, then -4, then 0, then -1, then 1, then 2, then -3, then 1. Best from index 0 is 2. Starting at index 1: 1, -2, 2, 1, 3, 4, -1, 3. Best from index 1 is 4.

## ALTERNATIVE:ADOPTED | explicita
Alternatively, Kadane's algorithm finds the answer in a single pass: at each element, the best subarray ending there is either the element alone or the element added to the best subarray ending at the previous position. That is far less work than nine starting points with up to nine sums each, so I will switch to it and abandon the enumeration.

## PROGRESSION:ADVANCES
Running Kadane's algorithm, with "best ending here" updated as max(x, previous + x): at -2 it is -2; at 1 it is 1; at -3 it is -2; at 4 it is 4; at -1 it is 3; at 2 it is 5; at 1 it is 6; at -5 it is 1; at 4 it is 5.

## PROGRESSION:ADVANCES
The largest value of "best ending here" is 6, reached at the element 1 at index 6. The subarray that achieves it is [4, -1, 2, 1].

## SUPPORT
The rule works because a running sum that has turned negative can only lower the total of any subarray that extends it, so it is better to start again at the current element.

## PROGRESSION:REACHES_OUTPUT
Thus final answer: 6.
