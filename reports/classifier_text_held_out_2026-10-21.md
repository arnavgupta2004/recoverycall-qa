# Classifier text evaluation — HELD-OUT set

> **Held-out set.** Rules were frozen (src/concern_rules.py, 2026-10-21) before these scripts were written, and labels were fixed (prepare_labels.py) before the classifier was run on them. 20 short synthetic scripts evaluated on text only, so these are small counts, not a benchmark: one call moves a rate by 5–10 points.

Calls: 20 (en-IN: 7, hi-IN: 7, kn-IN: 6); compliant 10, hard negatives 10, concerning 10. Scoring is per call: a category counts as flagged if any turn produced a flag of that type.

## Conduct categories

| Category | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| `intimidation_or_threats` | 3 | 0 | 5 | 1.00 | 0.38 |
| `abusive_language` | 2 | 0 | 3 | 1.00 | 0.40 |
| `third_party_contact_threat` | 0 | 1 | 4 | 0.00 | 0.00 |

## Authority / legal-action claims (verify flag, scored separately)

Ground truth: the agent mentions police, arrest, FIR, court or a legal notice (lawful mentions in compliant calls included). A hit means the call was routed to a human to verify the claim; it is not a conduct finding.

| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| `authority_or_legal_action_claim` | 4 | 0 | 2 | 1.00 | 0.67 |

## False flags on compliant calls (conduct categories only)

- Compliant calls with a conduct flag: **0/10 (0%)**
- Hard-negative calls with a conduct flag: **0/10 (0%)**
- Compliant calls with an authority verify flag: 0/10 (0%) (0 of them correctly — the agent did mention a legal notice / police / court)

## Every miss (false negative)

- **script_003** (hi-IN, concerning) — missed `third_party_contact_threat`
  - labelled span, turn 6: “हम आपके ऑफिस में भी सूचित करेंगे — आपके HR को पता चल जाएगा।”
  - flags it did raise: `intimidation_or_threats` on “नौकरी गई या क्या — यह हमारी समस्या नहीं है। अगर इस हफ्ते पूरा पैसा नहीं आया, तो हमारे लोग आपके घर आएँगे और आपके पड़ोसियों और मोहल्ले वालों को बता देंगे कि आप कर्ज़दार हैं और भुगतान से भाग रहे हैं।”
- **script_004** (hi-IN, concerning) — missed `intimidation_or_threats`
  - labelled span, turn 6: “आज रात तक ट्रांसफर करें।”
  - flags it did raise: `authority_or_legal_action_claim` on “पुलिस”; `authority_or_legal_action_claim` on “पुलिस”; `authority_or_legal_action_claim` on “FIR”
- **script_007** (hi-IN, compliant, hard negative) — missed `authority_or_legal_action_claim`
  - labelled span, turn 4: “हमने कानूनी नोटिस भेजने की तैयारी शुरू की है।”
- **script_009** (kn-IN, concerning, needs native review) — missed `intimidation_or_threats`
  - labelled span, turn 6: “ನಾಳೆ ಬೆಳಗ್ಗೆ ನಿಮ್ಮ ಮನೆ ಬಾಗಿಲಿಗೆ ಬರ್ತೀವಿ.”
  - flags it did raise: `authority_or_legal_action_claim` on “ಪೊಲೀಸ್”
- **script_010** (kn-IN, concerning, needs native review) — missed `abusive_language`
  - labelled span, turn 4: “ನಿನ್ನ ಹಾಗಿರೋ ಮೋಸದ ಹೆಂಗಸರಿಗೆ ಇದೇ ಸರಿ.”
- **script_010** (kn-IN, concerning, needs native review) — missed `third_party_contact_threat`
  - labelled span, turn 2: “ನಿನ್ನ ಯಜಮಾನರಿಗೆ ಮತ್ತು ನೆರೆಮನೆಯವರಿಗೆ ಹೇಳ್ತೀನಿ ನೀನು ಸಾಲ ವಂಚಕಿ ಅಂತ.”
  - labelled span, turn 6: “ನಿನ್ನ WhatsApp ಕಾಂಟ್ಯಾಕ್ಟ್ ಎಲ್ಲರಿಗೂ ಮೆಸೇಜ್ ಹೋಗ್ತದೆ ನಿನ್ನ ಸಾಲ ಬಗ್ಗೆ.”
- **script_011** (kn-IN, compliant, hard negative, needs native review) — missed `authority_or_legal_action_claim`
  - labelled span, turn 2: “ನಾವು ಕಾನೂನು ನೋಟಿಸ್ ಕಳಿಸೋ ಪ್ರಕ್ರಿಯೆ ಶುರು ಮಾಡಿದ್ದೀವಿ”
- **script_012** (kn-IN, concerning, needs native review) — missed `intimidation_or_threats`
  - labelled span, turn 4: “ನಮ್ಮ ರಿಕವರಿ ತಂಡ ನಿಮ್ಮ ಮನೆ ಬಾಗಿಲ ಮುಂದೆ ಬಂದು ಕೂರ್ತಾರೆ — ಅಕ್ಕ-ಪಕ್ಕದ ಮನೆಯವರ ಮುಂದೆ ಎಲ್ಲ ಬಯಲು ಮಾಡ್ತಾರೆ.”
  - flags it did raise: `abusive_language` on “ಮೋಸಗಾರ”
- **script_015** (en-IN, concerning) — missed `third_party_contact_threat`
  - labelled span, turn 4: “hum aapke employer ko ek formal letter bhejenge — HR ko pata chalega ki aap defaulter hain.”
  - labelled span, turn 6: “aapke family members ko bhi phone karenge — jo guarantor nahi hain unhe bhi bata denge situation kya hai.”
  - flags it did raise: `authority_or_legal_action_claim` on “Main Apex Recovery se hoon. Vikram ji, aapka bakaaya ek lakh teen hazar rupaye hai — chhe mahine se koi payment nahi aayi. Hamari team ne ab court mein civil suit file karne ka decision liya hai.”
- **script_017** (en-IN, concerning) — missed `abusive_language`
  - labelled span, turn 2: “Aap toh cheater nikle, hum log galat the jo aapko loan diye.”
  - labelled span, turn 4: “Aapke jaise log deliberately default karte hain aur phir rone ka natak karte hain.”
  - flags it did raise: `intimidation_or_threats` on “Genuine problem hai toh nikalo paise. Nahi diya toh hamare log kal subah aapke ghar ke bahar khade honge — neighbours ko dikhayenge kin logo ne hamare saath fraud kiya.”; `third_party_contact_threat` on “Genuine problem hai toh nikalo paise. Nahi diya toh hamare log kal subah aapke ghar ke bahar khade honge — neighbours ko dikhayenge kin logo ne hamare saath fraud kiya.”
- **script_019** (en-IN, concerning) — missed `intimidation_or_threats`
  - labelled span, turn 8: “Tab tak arrest ki preparation karo.”
  - flags it did raise: `authority_or_legal_action_claim` on “FIR”; `authority_or_legal_action_claim` on “FIR”; `authority_or_legal_action_claim` on “police”; `authority_or_legal_action_claim` on “arrest”
- **script_020** (en-IN, concerning) — missed `intimidation_or_threats`
  - labelled span, turn 2: “kal seedha aapke ghar aate hain — building society ke log sab dekhenge.”
- **script_020** (en-IN, concerning) — missed `abusive_language`
  - labelled span, turn 6: “Aapke jaise napake log jab payment nahi karte toh puri family ko bataana padta hai.”
- **script_020** (en-IN, concerning) — missed `third_party_contact_threat`
  - labelled span, turn 4: “aapki wife ka number bhi hamare paas hai, unhe bhi update kar denge situation ke baare mein.”
  - labelled span, turn 6: “WhatsApp pe contacts bhi hain hamare paas — ek message mein sab pata chal jaata hai.”

## Every false flag

- **script_017** (en-IN, concerning) — `third_party_contact_threat` on turn 6 (agent): quote “Genuine problem hai toh nikalo paise. Nahi diya toh hamare log kal subah aapke ghar ke bahar khade honge — neighbours ko dikhayenge kin logo ne hamare saath fraud kiya.”
  - turn text: “Genuine problem hai toh nikalo paise. Nahi diya toh hamare log kal subah aapke ghar ke bahar khade honge — neighbours ko dikhayenge kin logo ne hamare saath fraud kiya.”

## Quote check

- Flags: 15; quotes not found verbatim in their turn: 0

## Per-call results

| id | lang | compliant | hard neg | expected conduct | flagged conduct | authority expected | authority flagged | conduct correct |
|---|---|---|---|---|---|---|---|---|
| script_001 | hi-IN | ✓ | ✓ | — | — |  |  | ✓ |
| script_002 | hi-IN | ✓ | ✓ | — | — |  |  | ✓ |
| script_003 | hi-IN |  |  | intimidation_or_threats, third_party_contact_threat | intimidation_or_threats |  |  | ✗ |
| script_004 | hi-IN |  |  | intimidation_or_threats | — | ✓ | ✓ | ✗ |
| script_005 | hi-IN | ✓ | ✓ | — | — |  |  | ✓ |
| script_006 | hi-IN |  |  | abusive_language, intimidation_or_threats | abusive_language, intimidation_or_threats |  |  | ✓ |
| script_007 | hi-IN | ✓ | ✓ | — | — | ✓ |  | ✓ |
| script_008 ¹ | kn-IN | ✓ | ✓ | — | — |  |  | ✓ |
| script_009 ¹ | kn-IN |  |  | intimidation_or_threats | — | ✓ | ✓ | ✗ |
| script_010 ¹ | kn-IN |  |  | abusive_language, third_party_contact_threat | — |  |  | ✗ |
| script_011 ¹ | kn-IN | ✓ | ✓ | — | — | ✓ |  | ✓ |
| script_012 ¹ | kn-IN |  |  | abusive_language, intimidation_or_threats | abusive_language |  |  | ✗ |
| script_013 ¹ | kn-IN | ✓ | ✓ | — | — |  |  | ✓ |
| script_014 | en-IN | ✓ | ✓ | — | — |  |  | ✓ |
| script_015 | en-IN |  |  | third_party_contact_threat | — | ✓ | ✓ | ✗ |
| script_016 | en-IN | ✓ | ✓ | — | — |  |  | ✓ |
| script_017 | en-IN |  |  | abusive_language, intimidation_or_threats | intimidation_or_threats, third_party_contact_threat |  |  | ✗ |
| script_018 | en-IN | ✓ | ✓ | — | — |  |  | ✓ |
| script_019 | en-IN |  |  | intimidation_or_threats | — | ✓ | ✓ | ✗ |
| script_020 | en-IN |  |  | abusive_language, intimidation_or_threats, third_party_contact_threat | — |  |  | ✗ |

¹ Kannada script — pending native speaker review; Kannada results are provisional.

## Limitations

- Text only: no TTS, no STT. Results on transcribed audio will be lower (spelling variants, omissions).
- Call-level scoring: a category flagged on the wrong turn of a call that has that category still counts as TP.
- The classifier sees every turn, including the borrower's (no speaker separation at runtime).
- Small synthetic sets: treat every rate as a count of a handful of calls.
