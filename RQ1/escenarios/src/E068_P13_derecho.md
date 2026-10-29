---
id: E068
plantilla: P13
dominio: derecho
estilo: qwen
respuesta: B
enunciado: At a trial arising from a car accident, a witness for the plaintiff testifies: "The day after the accident, my neighbor told me that she had seen the defendant drive through the red light." The neighbor does not testify. The plaintiff offers the testimony to prove that the defendant ran the red light, and the defendant objects. How should the court rule? Options: A. Admit it as a present sense impression B. Exclude it as hearsay not within any exception C. Admit it because the witness is subject to cross-examination D. Admit it as a statement of a party opponent
---
## PROGRESSION:ADVANCES
Okay, let's see. The witness repeats in court what her neighbor told her the day after the accident: that the defendant ran the red light. It's offered to prove that he ran the light. Is it admissible?

## ALTERNATIVE:REJECTED | explicita
Alternatively, maybe there's no hearsay problem because the witness is on the stand and can be cross-examined. But the person whose perception matters is the neighbor, and she isn't in court; cross-examining the witness can't test what the neighbor saw. So I'll drop that idea.

## PROGRESSION:ADVANCES
The neighbor's statement was made out of court and is offered to prove the truth of what it asserts, that the defendant ran the light. That makes it hearsay, inadmissible unless an exception applies.

## ERROR | id=e1
The neighbor described something she saw, so the statement fits the present sense impression exception and is admissible.

## VERIFICATION:REFUTED | explicita
Wait, let me check the timing requirement of that exception. A present sense impression must be made while the declarant perceives the event or immediately afterward. The neighbor spoke the day after the accident, so the exception doesn't fit.

## CORRECTION:RECOVERED | explicita | id=c1
Correcting it: no exception covers a description given a day later, so the statement is inadmissible hearsay and the objection should be sustained.

## VERIFICATION:CONFIRMED | explicita | objetivo=c1
Let me double-check that conclusion against the purpose of the rule: the neighbor cannot be cross-examined about what she saw, which is exactly the danger the hearsay rule guards against, so excluding her statement is the right result. The conclusion stands.

## PROGRESSION:REACHES_OUTPUT
So the answer should be B: exclude it as hearsay not within any exception.
