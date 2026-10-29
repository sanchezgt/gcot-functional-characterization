---
id: E007
plantilla: P06
dominio: logica
estilo: qwen
respuesta: Cal
enunciado: Three friends, Ana, Ben, and Cal, each own exactly one pet: a dog, a cat, or a fish, with no two friends owning the same kind of pet. Ana is allergic to animal fur. Ben's pet barks. Who owns the cat?
---
## PROGRESSION:ADVANCES
Okay, let's see. Three friends, three pets, one each, no repeats: a dog, a cat, and a fish. Ana is allergic to fur. Ben's pet barks. I need to find who has the cat.

## ALTERNATIVE:REJECTED | implicita
Listing all six ways to hand out the three pets and striking out those that break a clue would find the cat's owner, but each clue fixes one friend's pet directly, so two steps are enough.

## PROGRESSION:ADVANCES
Ana is allergic to fur, and both the dog and the cat have fur, so Ana owns the fish.

## PROGRESSION:ADVANCES
Ben's pet barks, and the only pet that barks is the dog. So Ben owns the dog.

## PROGRESSION:ADVANCES
With the fish and the dog taken, the cat must belong to the remaining friend, Cal.

## PROGRESSION:REACHES_OUTPUT
So the answer should be Cal.
