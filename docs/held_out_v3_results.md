# Held-out v3 — end-to-end audio evaluation (run once)

18 externally written calls (6 per language: 3 concerning, 3 compliant; labels checked by reading and committed in 9ff7cdc before any audio was made). Pipeline per call: Gnani Timbre v2.5 TTS per turn → concatenation → **phone_clean** (300–3400 Hz, 8 kHz, G.711 μ-law) and **phone_noisy10** (the same plus band-limited pink noise at 10 dB SNR) → the app's `src.pipeline.analyse`: energy VAD segments ≤ 12 s → Gnani Prisma STT → coverage check + re-split recovery → frozen rules v1 / Gemini tagger with quote verification / hybrid. LLM backend: `gemini:gemini-3.5-flash-lite`. Run time 534 s.

The six concerning/compliant situations are each written once per language (HI00k ≈ KN00k ≈ HX00k), so per-language numbers are three renderings of the same six situations, not independent samples. Kannada scripts await native-speaker review. Every number below is a count over a handful of calls.

## Summary

| variant | method | conduct TP/FP/FN | compliant false flags | hard-negative false flags | authority (verify) TP/FP/FN |
|---|---|---|---|---|---|
| phone_clean | rules | 0/0/15 | 0/9 | 0/9 | 2/0/4 |
| phone_clean | llm | 15/0/0 | 0/9 | 0/9 | 6/0/0 |
| phone_clean | hybrid | 15/0/0 | 0/9 | 0/9 | 6/0/0 |
| phone_noisy10 | rules | 0/0/15 | 0/9 | 0/9 | 2/0/4 |
| phone_noisy10 | llm | 15/0/0 | 0/9 | 0/9 | 6/1/0 |
| phone_noisy10 | hybrid | 15/0/0 | 0/9 | 0/9 | 6/1/0 |

## Per-category TP / FP / FN per language

Call-level: a category counts as flagged when any segment of the call produced a flag of that type. Conduct = intimidation, abusive language, third-party contact; authority claims are a separate verify flag scored against the agent's mention of police / arrest / FIR / court / legal notice.

### phone_clean

| category | language | rules TP/FP/FN | llm TP/FP/FN | hybrid TP/FP/FN |
|---|---|---|---|---|
| `intimidation_or_threats` | Hindi | 0/0/2 | 2/0/0 | 2/0/0 |
| `intimidation_or_threats` | Kannada | 0/0/2 | 2/0/0 | 2/0/0 |
| `intimidation_or_threats` | Hinglish | 0/0/2 | 2/0/0 | 2/0/0 |
| `abusive_language` | Hindi | 0/0/1 | 1/0/0 | 1/0/0 |
| `abusive_language` | Kannada | 0/0/1 | 1/0/0 | 1/0/0 |
| `abusive_language` | Hinglish | 0/0/1 | 1/0/0 | 1/0/0 |
| `third_party_contact_threat` | Hindi | 0/0/2 | 2/0/0 | 2/0/0 |
| `third_party_contact_threat` | Kannada | 0/0/2 | 2/0/0 | 2/0/0 |
| `third_party_contact_threat` | Hinglish | 0/0/2 | 2/0/0 | 2/0/0 |
| `authority_or_legal_action_claim` (verify flag) | Hindi | 1/0/1 | 2/0/0 | 2/0/0 |
| `authority_or_legal_action_claim` (verify flag) | Kannada | 0/0/2 | 2/0/0 | 2/0/0 |
| `authority_or_legal_action_claim` (verify flag) | Hinglish | 1/0/1 | 2/0/0 | 2/0/0 |
| **conduct total** | all | **0/0/15** | **15/0/0** | **15/0/0** |

### phone_noisy10

| category | language | rules TP/FP/FN | llm TP/FP/FN | hybrid TP/FP/FN |
|---|---|---|---|---|
| `intimidation_or_threats` | Hindi | 0/0/2 | 2/0/0 | 2/0/0 |
| `intimidation_or_threats` | Kannada | 0/0/2 | 2/0/0 | 2/0/0 |
| `intimidation_or_threats` | Hinglish | 0/0/2 | 2/0/0 | 2/0/0 |
| `abusive_language` | Hindi | 0/0/1 | 1/0/0 | 1/0/0 |
| `abusive_language` | Kannada | 0/0/1 | 1/0/0 | 1/0/0 |
| `abusive_language` | Hinglish | 0/0/1 | 1/0/0 | 1/0/0 |
| `third_party_contact_threat` | Hindi | 0/0/2 | 2/0/0 | 2/0/0 |
| `third_party_contact_threat` | Kannada | 0/0/2 | 2/0/0 | 2/0/0 |
| `third_party_contact_threat` | Hinglish | 0/0/2 | 2/0/0 | 2/0/0 |
| `authority_or_legal_action_claim` (verify flag) | Hindi | 1/0/1 | 2/0/0 | 2/0/0 |
| `authority_or_legal_action_claim` (verify flag) | Kannada | 0/0/2 | 2/0/0 | 2/0/0 |
| `authority_or_legal_action_claim` (verify flag) | Hinglish | 1/0/1 | 2/1/0 | 2/1/0 |
| **conduct total** | all | **0/0/15** | **15/0/0** | **15/0/0** |

## False flags on compliant and hard-negative calls (conduct categories only)

| variant | language | rules compliant | rules hard-neg | llm compliant | llm hard-neg | hybrid compliant | hybrid hard-neg |
|---|---|---|---|---|---|---|---|
| phone_clean | Hindi | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| phone_clean | Kannada | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| phone_clean | Hinglish | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| phone_clean | **all** | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |
| phone_noisy10 | Hindi | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| phone_noisy10 | Kannada | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| phone_noisy10 | Hinglish | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| phone_noisy10 | **all** | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |

Authority verify flags on compliant calls (expected where the agent mentions a legal notice; not conduct false flags): 6 across both variants.

## Quote verification

| variant | LLM quotes rejected | flags dropped after merge |
|---|---|---|
| phone_clean | 0 of 22 LLM outputs | 0 |
| phone_noisy10 | 0 of 23 LLM outputs | 0 |


LLM step not ok (rules-only fallback, no LLM flags): 0 of 36 analyses.

## Omissions: detected and recovered

A turn is omitted when its best word-window match against the first-pass STT text of the overlapping segments is below 0.5 (≤ 3-word turns need 0.8). Detected = ≥ 50 % of the turn lies in a segment the coverage check flagged. Recovered = present after re-split recovery. Still missing = absent from the final transcript; surfaced = it overlaps an 'untranscribed speech' warning shown to the reviewer.

| variant | language | turns | omitted | detected | recovered | still missing (surfaced) | segments flagged |
|---|---|---|---|---|---|---|---|
| phone_clean | Hindi | 36 | 1 | 0 | 0 | 1 (0) | 0/17 |
| phone_clean | Kannada | 36 | 0 | 0 | 0 | 0 (0) | 0/18 |
| phone_clean | Hinglish | 36 | 0 | 0 | 0 | 0 (0) | 0/18 |
| phone_clean | **all** | 108 | 1 | 0 | 0 | 1 (0) | 0/53 |
| phone_noisy10 | Hindi | 36 | 0 | 0 | 0 | 0 (0) | 0/17 |
| phone_noisy10 | Kannada | 36 | 0 | 0 | 0 | 0 (0) | 0/18 |
| phone_noisy10 | Hinglish | 36 | 0 | 0 | 0 | 0 (0) | 0/18 |
| phone_noisy10 | **all** | 108 | 0 | 0 | 0 | 0 (0) | 0/53 |

- phone_clean HI002 turn 1 (customer, 12 words): match 0.42 → 0.42; not detected; still missing, silent

## rules: every miss and false flag

- phone_clean MISS **HI001** `intimidation_or_threats` — “आज पैसे नहीं आए तो हमारे लोग आपके घर आकर ऐसा समझाएँगे कि याद रखोगे।”
- phone_clean MISS **HI001** `third_party_contact_threat` — “फिर अपने घरवालों को भी तैयार रखना, सबको पता चल जाएगा कि आपने कर्ज़ लिया है और वापस नहीं कर रहे।”
- phone_clean MISS **HI002** `intimidation_or_threats` — “समय खत्म होने पर हमारी टीम खुद आपसे मिलेगी। फिर मत कहना कि बताया नहीं था।”
- phone_clean MISS **HI002** `abusive_language` — “आप जैसे निकम्मे लोग कर्ज़ लेकर गायब होने के लिए ही होते हैं।”
- phone_clean MISS **HI003** `third_party_contact_threat` — “आपके दफ्तर में भी फोन करके बता देंगे कि आप कर्ज़ नहीं चुका रहे। फिर आपकी नौकरी बचाना।”
- phone_clean MISS **HI004** `authority_or_legal_action_claim` — “यदि भुगतान नहीं होता है, तो कंपनी की नीति और लागू प्रक्रिया के अनुसार कानूनी नोटिस भेजा जा सकता है। यह अभी केवल प्रक्रिया की जानकारी है।”
- phone_clean MISS **KN001** `intimidation_or_threats` — “ಇವತ್ತು ಹಣ ಬರದಿದ್ರೆ ನಮ್ಮವರು ಮನೆಗೆ ಬಂದು ಮಾತಾಡ್ತಾರೆ, ಆಮೇಲೆ ನಿಮಗೇ ಗೊತ್ತಾಗುತ್ತೆ.”
- phone_clean MISS **KN001** `third_party_contact_threat` — “ಹಾಗಾದ್ರೆ ನಿಮ್ಮ ಅಕ್ಕಪಕ್ಕದವರಿಗೂ ಮನೆಯವರಿಗೂ ನಿಮ್ಮ ಸಾಲದ ವಿಷಯ ಹೇಳಬೇಕಾಗುತ್ತೆ.”
- phone_clean MISS **KN002** `intimidation_or_threats` — “ಇನ್ನೂ ಕಾಯಿಸಬೇಡಿ. ನಮ್ಮ ತಂಡ ನಿಮ್ಮನ್ನು ಹುಡುಕಿಕೊಂಡು ಬರುತ್ತೆ, ಆಗ ಏನು ಮಾಡ್ತೀರೋ ನೋಡೋಣ.”
- phone_clean MISS **KN002** `abusive_language` — “ನೀವು ಒಬ್ಬ ಬೇಜವಾಬ್ದಾರಿ ಮನುಷ್ಯ.”
- phone_clean MISS **KN003** `third_party_contact_threat` — “ನಿಮ್ಮ ಆಫೀಸಿಗೂ ಫೋನ್ ಮಾಡಿ ನೀವು ಸಾಲ ತೀರಿಸಿಲ್ಲ ಅಂತ ಎಲ್ಲರಿಗೂ ಹೇಳುತ್ತೇವೆ.”
- phone_clean MISS **KN003** `authority_or_legal_action_claim` — “ತಕ್ಷಣ ಹಣದ ಸಾಕ್ಷಿ ಕಳಿಸಿ, ಇಲ್ಲದಿದ್ದರೆ ನಿಮ್ಮ ಮೇಲೆ FIR ಹಾಕಿಸಿ ಪೊಲೀಸರಿಂದ ಕ್ರಮ ತೆಗೆಸುತ್ತೇವೆ.”
- phone_clean MISS **KN004** `authority_or_legal_action_claim` — “ಪಾವತಿ ಆಗದಿದ್ದರೆ, ಅನ್ವಯಿಸುವ ನಿಯಮಗಳು ಮತ್ತು ಸರಿಯಾದ ಪ್ರಕ್ರಿಯೆಯ ಪ್ರಕಾರ ಕಾನೂನು ನೋಟಿಸ್ ಕಳಿಸುವ ಸಾಧ್ಯತೆ ಇದೆ.”
- phone_clean MISS **HX001** `intimidation_or_threats` — “Aaj payment nahi aaya toh hamare log aapke ghar aake aapko achhe se samjhayenge.”
- phone_clean MISS **HX001** `third_party_contact_threat` — “Toh apni family ko bhi bata dena, warna hum khud sabko aapke loan ke baare mein bata denge.”
- phone_clean MISS **HX002** `intimidation_or_threats` — “Time khatam hua toh hamari team khud aapko dhoondhne niklegi. Phir complaint mat karna.”
- phone_clean MISS **HX002** `abusive_language` — “Aap jaise irresponsible log loan lete waqt bade promises karte ho aur baad mein nautanki karte ho.”
- phone_clean MISS **HX003** `third_party_contact_threat` — “Aapke manager ko bhi call karke bata denge ki aap loan default kar rahe ho. Phir job ka soch lena.”
- phone_clean MISS **HX004** `authority_or_legal_action_claim` — “Agar dues unpaid rehte hain, toh applicable process ke according legal notice issue ho sakta hai. Yeh process ki factual information hai.”
- phone_noisy10 MISS **HI001** `intimidation_or_threats` — “आज पैसे नहीं आए तो हमारे लोग आपके घर आकर ऐसा समझाएँगे कि याद रखोगे।”
- phone_noisy10 MISS **HI001** `third_party_contact_threat` — “फिर अपने घरवालों को भी तैयार रखना, सबको पता चल जाएगा कि आपने कर्ज़ लिया है और वापस नहीं कर रहे।”
- phone_noisy10 MISS **HI002** `intimidation_or_threats` — “समय खत्म होने पर हमारी टीम खुद आपसे मिलेगी। फिर मत कहना कि बताया नहीं था।”
- phone_noisy10 MISS **HI002** `abusive_language` — “आप जैसे निकम्मे लोग कर्ज़ लेकर गायब होने के लिए ही होते हैं।”
- phone_noisy10 MISS **HI003** `third_party_contact_threat` — “आपके दफ्तर में भी फोन करके बता देंगे कि आप कर्ज़ नहीं चुका रहे। फिर आपकी नौकरी बचाना।”
- phone_noisy10 MISS **HI004** `authority_or_legal_action_claim` — “यदि भुगतान नहीं होता है, तो कंपनी की नीति और लागू प्रक्रिया के अनुसार कानूनी नोटिस भेजा जा सकता है। यह अभी केवल प्रक्रिया की जानकारी है।”
- phone_noisy10 MISS **KN001** `intimidation_or_threats` — “ಇವತ್ತು ಹಣ ಬರದಿದ್ರೆ ನಮ್ಮವರು ಮನೆಗೆ ಬಂದು ಮಾತಾಡ್ತಾರೆ, ಆಮೇಲೆ ನಿಮಗೇ ಗೊತ್ತಾಗುತ್ತೆ.”
- phone_noisy10 MISS **KN001** `third_party_contact_threat` — “ಹಾಗಾದ್ರೆ ನಿಮ್ಮ ಅಕ್ಕಪಕ್ಕದವರಿಗೂ ಮನೆಯವರಿಗೂ ನಿಮ್ಮ ಸಾಲದ ವಿಷಯ ಹೇಳಬೇಕಾಗುತ್ತೆ.”
- phone_noisy10 MISS **KN002** `intimidation_or_threats` — “ಇನ್ನೂ ಕಾಯಿಸಬೇಡಿ. ನಮ್ಮ ತಂಡ ನಿಮ್ಮನ್ನು ಹುಡುಕಿಕೊಂಡು ಬರುತ್ತೆ, ಆಗ ಏನು ಮಾಡ್ತೀರೋ ನೋಡೋಣ.”
- phone_noisy10 MISS **KN002** `abusive_language` — “ನೀವು ಒಬ್ಬ ಬೇಜವಾಬ್ದಾರಿ ಮನುಷ್ಯ.”
- phone_noisy10 MISS **KN003** `third_party_contact_threat` — “ನಿಮ್ಮ ಆಫೀಸಿಗೂ ಫೋನ್ ಮಾಡಿ ನೀವು ಸಾಲ ತೀರಿಸಿಲ್ಲ ಅಂತ ಎಲ್ಲರಿಗೂ ಹೇಳುತ್ತೇವೆ.”
- phone_noisy10 MISS **KN003** `authority_or_legal_action_claim` — “ತಕ್ಷಣ ಹಣದ ಸಾಕ್ಷಿ ಕಳಿಸಿ, ಇಲ್ಲದಿದ್ದರೆ ನಿಮ್ಮ ಮೇಲೆ FIR ಹಾಕಿಸಿ ಪೊಲೀಸರಿಂದ ಕ್ರಮ ತೆಗೆಸುತ್ತೇವೆ.”
- phone_noisy10 MISS **KN004** `authority_or_legal_action_claim` — “ಪಾವತಿ ಆಗದಿದ್ದರೆ, ಅನ್ವಯಿಸುವ ನಿಯಮಗಳು ಮತ್ತು ಸರಿಯಾದ ಪ್ರಕ್ರಿಯೆಯ ಪ್ರಕಾರ ಕಾನೂನು ನೋಟಿಸ್ ಕಳಿಸುವ ಸಾಧ್ಯತೆ ಇದೆ.”
- phone_noisy10 MISS **HX001** `intimidation_or_threats` — “Aaj payment nahi aaya toh hamare log aapke ghar aake aapko achhe se samjhayenge.”
- phone_noisy10 MISS **HX001** `third_party_contact_threat` — “Toh apni family ko bhi bata dena, warna hum khud sabko aapke loan ke baare mein bata denge.”
- phone_noisy10 MISS **HX002** `intimidation_or_threats` — “Time khatam hua toh hamari team khud aapko dhoondhne niklegi. Phir complaint mat karna.”
- phone_noisy10 MISS **HX002** `abusive_language` — “Aap jaise irresponsible log loan lete waqt bade promises karte ho aur baad mein nautanki karte ho.”
- phone_noisy10 MISS **HX003** `third_party_contact_threat` — “Aapke manager ko bhi call karke bata denge ki aap loan default kar rahe ho. Phir job ka soch lena.”
- phone_noisy10 MISS **HX004** `authority_or_legal_action_claim` — “Agar dues unpaid rehte hain, toh applicable process ke according legal notice issue ho sakta hai. Yeh process ki factual information hai.”

## llm: every miss and false flag

- phone_noisy10 FALSE VERIFY FLAG **HX002** `authority_or_legal_action_claim` (no agent authority mention in the script): “phir complaint baat karenge”

## hybrid: every miss and false flag

- phone_noisy10 FALSE VERIFY FLAG **HX002** `authority_or_legal_action_claim` (no agent authority mention in the script): “phir complaint baat karenge”

## API calls used

- Gnani Timbre TTS: 108 requests (0 turns from cache)
- Gnani Prisma STT: 106 requests incl. recovery pieces (0 from cache)
- Gemini: 34 requests incl. retries for 36 analyses (an analysis whose transcript is identical to an earlier one is answered from the LLM cache); non-200 attempts: 0
- Failed requests after client retries: 0

## Limitations

- Synthetic TTS speech, one noise type, one seed per call; real calls have overlap, accents, crosstalk and real line noise.
- 18 calls = 6 situations × 3 languages; Kannada awaits native review; Hinglish is romanised and voiced by a hi-en TTS voice.
- Call-level scoring: a flag of the right type anywhere in a call counts as a TP.
- No speaker separation: rules and the LLM see both speakers' words.
- The coverage baseline (chars per speech-second) was measured on the spike's clean audio, not on this set.
