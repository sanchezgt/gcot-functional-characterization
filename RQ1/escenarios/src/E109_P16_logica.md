---
id: E109
plantilla: P16
dominio: logica
estilo: deepseek
respuesta: 3
enunciado: A drawer contains 10 red socks and 10 blue socks, all mixed together. In complete darkness, what is the smallest number of socks you must take out to be certain of having at least two socks of the same color?
---
## PROGRESSION:ADVANCES
We are given: "A drawer contains 10 red socks and 10 blue socks. In the dark, what is the smallest number of socks you must take out to be certain of having two of the same color?"

We need the minimum number of draws that guarantees a matching pair, whatever socks come out.

## PROGRESSION:ABANDONED
I start listing the possible draw sequences. With two socks the outcomes are red-red, red-blue, blue-red, and blue-blue. With three socks they are red-red-red, red-red-blue, red-blue-red, red-blue-blue, blue-red-red...

## ALTERNATIVE:ADOPTED | implicita
Only the colors matter, and there are just two of them, so each sock drawn can be treated as an item placed into one of two color boxes. By the pigeonhole principle, as soon as there are more items than boxes, some box holds two items, so the question becomes how many socks exceed two boxes.

## PROGRESSION:ADVANCES
Two socks can be one red and one blue, with no pair. A third sock must go into the red box or the blue box, and either way that box then holds two socks. So 3 socks guarantee a pair, while 2 do not.

## SUPPORT
The 10 socks of each color play no role beyond making sure each color can supply two socks; the guarantee depends only on the number of colors.

## PROGRESSION:REACHES_OUTPUT
So the smallest number of socks to take out is 3.

Thus final answer: 3.
