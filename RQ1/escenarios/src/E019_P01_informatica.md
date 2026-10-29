---
id: E019
plantilla: P01
dominio: informatica
estilo: terso
respuesta: B
enunciado: What is the worst-case time complexity of binary search on a sorted array of n elements? Options: A. O(1) B. O(log n) C. O(n) D. O(n log n)
---
## PROGRESSION:ADVANCES
Need the worst-case complexity of binary search on a sorted array of n elements.

## PROGRESSION:ADVANCES
Binary search compares the target with the middle element and discards the half that cannot contain it. Each comparison halves the remaining range.

## PROGRESSION:ADVANCES
Starting from n elements, after k comparisons the range has about n / 2^k elements. The search ends when the range has at most 1 element, so n / 2^k <= 1, which gives k >= log2(n). Worst case: about log2(n) comparisons, so O(log n).

## VERIFICATION:CONFIRMED | explicita
Quick check with a concrete n: for n = 1024, the halving sequence is 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, which is 10 halvings, and log2(1024) = 10. Confirmed.

## SUPPORT
Each step does constant work (one comparison and an index update), so the number of steps is the whole cost.

## PROGRESSION:REACHES_OUTPUT
Worst case O(log n). Answer: B.
