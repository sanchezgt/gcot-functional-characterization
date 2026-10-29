---
id: E092
plantilla: P18
dominio: medicina
estilo: qwen
respuesta: B
enunciado: A 65-year-old man who smokes is brought to the physician because of 1 week of confusion. He appears euvolemic. Serum sodium is 125 mEq/L and serum osmolality is 260 mOsm/kg. Urine osmolality is 500 mOsm/kg and urine sodium is 45 mEq/L. Which of the following is the most likely cause of his hyponatremia? Options: A. Pseudohyponatremia B. Syndrome of inappropriate antidiuretic hormone secretion C. Primary polydipsia D. Hypovolemic hyponatremia
---
## PROGRESSION:ADVANCES
Okay, let's see. A 65-year-old smoker with confusion, serum sodium 125, serum osmolality 260, urine osmolality 500, urine sodium 45, and euvolemic on exam. I need the cause of the hyponatremia.

## ERROR | id=e1
First step is the serum osmolality. 260 is in the normal range, so this is hyponatremia with a normal osmolality.

## PROGRESSION:ADVANCES
Hyponatremia with a normal osmolality points to pseudohyponatremia, where very high lipids or proteins lower the measured sodium without changing the true tonicity of the plasma.

## PROGRESSION:ADVANCES
So the next thing to look for would be severe hypertriglyceridemia or a paraprotein, as in multiple myeloma.

## VERIFICATION:REFUTED | explicita | objetivo=e1
Hmm, wait. Let me check that first step again. Normal serum osmolality is about 275 to 295 mOsm/kg, so 260 is low, not normal. The pseudohyponatremia path was built on a misread value.

## CORRECTION:RECOVERED | explicita | objetivo=e1
Correcting it: this is hypotonic hyponatremia. With euvolemia, an inappropriately concentrated urine of 500 mOsm/kg, and a urine sodium of 45, it is the syndrome of inappropriate antidiuretic hormone secretion.

## SUPPORT
In a smoker, SIADH raises the possibility of small cell lung cancer, which can secrete antidiuretic hormone ectopically.

## PROGRESSION:REACHES_OUTPUT
So the answer should be B, SIADH.
