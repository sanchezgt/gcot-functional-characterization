---
id: E066
plantilla: P13
dominio: medicina
estilo: narrativo
respuesta: B
enunciado: A 22-year-old man with type 1 diabetes mellitus comes to the emergency department with 2 days of vomiting and abdominal pain. He is breathing deeply and rapidly. Serum studies show sodium 135 mEq/L, chloride 100 mEq/L, bicarbonate 10 mEq/L, and glucose 480 mg/dL; urine ketones are strongly positive. Which of the following best describes his acid-base disorder? Options: A. Normal anion gap metabolic acidosis B. High anion gap metabolic acidosis C. Metabolic alkalosis D. Respiratory acidosis
---
## PROGRESSION:ADVANCES
First, I gather the facts: a young man with type 1 diabetes, two days of vomiting and abdominal pain, deep rapid breathing, a bicarbonate of 10, a glucose of 480, and strongly positive urine ketones. I need to classify the acid-base disorder.

## ALTERNATIVE:REJECTED | explicita
Alternatively, the vomiting might point to a metabolic alkalosis, since losing gastric acid raises bicarbonate. But his bicarbonate is 10, far below normal, which is the opposite of an alkalosis, so I set that possibility aside.

## PROGRESSION:ADVANCES
A low bicarbonate with deep, rapid breathing is a metabolic acidosis with respiratory compensation. To classify it further, I calculate the anion gap.

## ERROR | id=e1
Anion gap = sodium - chloride + bicarbonate = 135 - 100 + 10 = 45 mEq/L.

## VERIFICATION:REFUTED | explicita
Wait, let me check that formula. The anion gap is sodium minus the sum of chloride and bicarbonate, so bicarbonate has to be subtracted, not added. The value of 45 comes from the wrong sign.

## CORRECTION:RECOVERED | explicita | id=c1
Correcting it: anion gap = 135 - (100 + 10) = 25 mEq/L.

## VERIFICATION:CONFIRMED | explicita | objetivo=c1
Let me double-check that number: 100 + 10 = 110, and 135 - 110 = 25. Confirmed.

## PROGRESSION:ADVANCES
A normal anion gap is about 12 ± 4, so 25 is clearly elevated. A high anion gap metabolic acidosis in a patient with type 1 diabetes, marked hyperglycemia, and ketonuria is diabetic ketoacidosis: the unmeasured anions are the ketoacids beta-hydroxybutyrate and acetoacetate.

## PROGRESSION:REACHES_OUTPUT
His disorder is a high anion gap metabolic acidosis, option B.

Final answer: B.
