# Results posts (drafts)

## X reply (258 characters)

> Results: on 18 held-out fictional calls (Hindi, Kannada, Hinglish) synthesized with Gnani Timbre and run through Prisma on phone + noisy audio, the hybrid flagged 15/15 labelled concerns, 0/9 false flags on lawful calls. Small synthetic set, not a benchmark.

## LinkedIn follow-up

A follow-up on RecoveryCall QA, my project for the Great Indian AI Internship Challenge. It's a review aid for debt-recovery calls in Hindi, Kannada and Hinglish, built on Gnani Prisma (speech-to-text) and Gnani Timbre (text-to-speech).

I ran one final end-to-end test on 18 fictional calls, written by someone else and labelled before anything ran on them. Each call was voiced with Timbre, degraded to phone quality (with and without 10 dB noise), and sent through the same pipeline as the live app: Prisma STT, a coverage check, then rules plus Gemini with verbatim-quote checking.

What came out:
• The hybrid flagged all 15 labelled concerns (intimidation, abuse, threats to tell family or employers) in all three languages, on both versions of the audio.
• It raised no conduct flags on the 9 compliant calls, which were deliberately firm: late fees, legal-notice process, settlement deadlines, guarantor clauses.
• Every AI quote matched the transcript word for word; none had to be rejected.
• The frozen keyword rules found none of the 15 on their own. The model does the heavy lifting, and the quote check keeps it honest.

What didn't go perfectly:
• One mis-transcribed Hinglish phrase turned into a false "verify this claim" flag.
• Prisma dropped one short borrower reply that my coverage check didn't catch. Short turns remain the blind spot.

Caveats: this is a small synthetic set (6 situations × 3 languages), the Kannada scripts still need a native speaker's review, and real calls will be harder. Treat it as a smoke test, not a benchmark.

Demo: https://recoverycall-qa-711436060077.asia-south1.run.app
Write-up: <ADD REPO LINK>/blob/main/docs/final_report.md

#GreatIndianAIInternshipChallenge
