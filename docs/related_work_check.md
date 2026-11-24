# Related-work check for the owner-forensics claim (2026-11-24)

Question (step 0 of the task): does any paper propose a defence that **restores the original
owner's claim after re-watermarking**, or **determines the embedding order**?

Method: all eight PDFs of section 1 of `docs/research_plan.md` were downloaded from arxiv.org and
read as extracted text (abstract, threat model, related work, limitations, keyword search for
overwrit/re-watermark/ownership/dispute/order/timestamp). Two web searches ("re-watermarking
defense ...", "watermark embedding order ...") then found further work; those papers were
downloaded and read the same way. Text extraction, not page images: tables and figures were not
inspected visually. Search was done with a general web search, not Google Scholar / Semantic
Scholar, so it is not exhaustive.

## Verdict

**The claim in the plan, "nobody has published a way for the rightful owner to get the claim back",
is too strong and must be narrowed.** Two training-time defences exist. What I did not find is a
**post-hoc, owner-side forensic protocol that works on already deployed open watermarks
(TrustMark, WAM, Video Seal) and decides the embedding order**. That narrower claim is supported
by everything I read, but only as "not found", not "does not exist".

| What | Prior art found | Status for our claim |
|---|---|---|
| Re-watermarking as a removal attack | Bulychev 2026; WARP 2026; Senthuran 2026; Jia 2025; Yao 2025 (audio) | not novel |
| **Ownership hijack by re-watermarking (attacker's message readable)** | Bulychev 2026, Table 2: attack message bit accuracy 0.99 cross-method, 0.78–0.99 same-method, on WAM, Video Seal, StegaStamp, RoSteALS, Pixel Seal, ZoDiac (DiffusionDB + COCO); Senthuran 2026 "transfers the ownership to an attacker" | **plan contribution 1 is an extension, not a new finding** (adds TrustMark, product photos, thief ΔE, strength sweep, JPEG, n + CI) |
| Defence that keeps the original message readable | **Senthuran et al., ICLR 2026**: Lipschitz-bounded encoder-decoder + re-watermarking adversarial training, with a per-bit recoverability bound; white-box thief who reuses the same encoder. **Jia et al. 2025** (arXiv 2508.17247): adversarial training for multi-embedding attacks on face-forensics watermarks | exists, but **embedder-side, needs retraining, not applicable to TrustMark/WAM/Video Seal as deployed**; tested on their own models only; Senthuran covers self-re-watermarking only (their limitation) |
| Post-hoc owner-side recovery on an open system | none found | **our gap** |
| Determining embedding order of two blind neural watermarks | none found (search: "no paper infers the sequence of embeddings"); GSF recovers per-region generation but only for one cooperative publisher | **our gap** |
| Resolving ownership with the original | Classical: Craver et al. 1997/1998 (invertibility, ambiguity attacks, non-invertible marks), timestamps; US patent 6,272,634 (two-watermark protocol); Mintzer–Braudaway 1999; Mascher-Kampfer 2006 | **the idea of resolving a dispute with the original and a timestamp is classical**; we must position the order proof as its instance for blind neural watermarks, not as a new principle |

## Paper by paper (read in full text unless noted)

| Paper | What it does | Defence / restores owner's claim / order? |
|---|---|---|
| **Bulychev, Marchant, Rubinstein**, arXiv 2605.16796 | 8 schemes (2 in-processing, 6 post-processing: StegaStamp, RoSteALS, ZoDiac, Pixel Seal, WAM, Video Seal), DiffusionDB + COCO; re-watermark, classifier names the scheme (accuracy 0.878–0.953). Secondary goal "ownership forgery", Table 2. | **No defence.** Only remark: the classifier can serve providers as an audit. Mitigation called "challenging". No order determination. TrustMark only cited. |
| **Yao et al.**, arXiv 2509.05835 (audio, AAAI 2026 line) | Overwriting attack on AudioSeal, Timbre, WavMark; describes the arbitration protocol "owner reveals m_owner, arbiter checks D(x_w) = m_owner". | **No defence**, calls for security in future designs. Audio. |
| **Senthuran et al.**, ICLR 2026 | Self-re-watermarking threat (same encoder reused); Lipschitz-constrained "self-aware" framework + adversarial training; bound on message recoverability. Metrics ACC_clean / ACC_adv / ACC_orig. | **Yes, a defence, but training-time, embedder-side, white-box self-re-watermarking only.** No industry system evaluated (no TrustMark/WAM/Video Seal in the text). No order, no registration, no third party. |
| **Jia et al.**, arXiv 2508.17247 | Multi-embedding attack on deepfake proactive-forensics watermarks; "Adversarial Interference Simulation" training paradigm. | Training-time defence; face forensics. No order. |
| **Dong, Li, Wang**, arXiv 2609.32241 | Residual transferability = forgery by copying residuals; CoverLock. | Forgery, not overwriting. No. (Relevant to adaptive attack A4 / our copy attack.) |
| **Wang et al.**, arXiv 2604.03693 (ResGuard) | Known-original attack (pairs → estimated residual → subtraction); image-dependent residual loss. | Embedder defence; no disputes. (Relevant to A4: estimating the original.) |
| **Nemecek et al.**, arXiv 2603.02378 | C2PA manifest vs watermark conflicts ("integrity clash"), cross-layer audit. | No ordering of two watermarks. **The plan's "CVPR 2026 Workshops, best paper" is not stated in the PDF; cite as arXiv.** |
| **Li et al.**, arXiv 2611.23363 (GSF) | Reissuance: current/superseded/unissued + per-region last authorized revision, authenticated live head. | Cooperative: one publisher, secret key. **Own limitation: a second GSF mark from a keyless party disrupts the issue ("loses its status").** Cites Bulychev as "overwrite attack". Not adversarial. |
| **Kazempourradi et al.**, arXiv 2610.04151 | Open-licence soft-binding benchmark. 12 methods. | No overwriting study. Useful numbers: **TrustMark's error-correcting detector fires on 5.9–15.3% of unmarked images** (consistent with our 2.4% "present" on 1,167 negatives; use a payload test), 50% of each side (25% area) centre crop → TrustMark and Video Seal at chance. |
| **Abud et al.**, WARP, arXiv 2609.40031 (ACM MM 2026) | 32 methods × 34 attacks, includes a victim × attacker re-embedding matrix (TPR@0.01% FPR); keyed methods survive different-key re-embedding; same method twice erases. | Measures the victim's survival, not the attacker's claim; **no defence, no order**. Overlaps with our measurement part. |
| PAI, arXiv 2601.06639 (found by search) | Inherent watermark for diffusion services, spoofing / removal / tamper localisation. | No re-watermarking. Not relevant. |
| WAVES (2024) | Not downloaded, not read. | – |
| Craver 1997/1998, Mintzer–Braudaway 1999, Mascher-Kampfer 2006, Sencar 2007, US 6,272,634 | **Not read**; known from the existing `references.bib` (Craver) or from search snippets / the GSF reference list. | Read the Craver paper before the thesis states how order evidence relates to the deadlock problem. |

## Consequences for the plan

1. Replace the gap sentence by: *"Existing defences against re-watermarking are training-time
   (Senthuran et al. 2026; Jia et al. 2025) and only protect models that were trained with them.
   Open industry watermarks (TrustMark, WAM, Video Seal) are deployed without such protection.
   We give the owner a post-hoc forensic protocol that needs no retraining."*
2. Contribution 1 (hijack measurement) becomes "extends Bulychev et al. and WARP to product
   photos, TrustMark, thief ΔE and strength, with intervals", not "first to show hijack".
3. The order proof is positioned as a modern instance of the classical rightful-ownership
   resolution (Craver et al.), applied to **blind neural** watermarks with a calibrated test.
4. Compare, where possible, against Senthuran et al. on the same attack (their code is public:
   github.com/SVithurabiman/SRW). Not done here (it would need their trained models; no training
   allowed) — state as a limitation.
5. Search was a general web search on 2026-11-24. Set the alerts the plan suggests and repeat
   before submission.

## Bibliography

Verified entries added to `thesis/references.bib` (full text read): bulychev2026watermarks,
yao2025overwriting, senthuran2026selfrewatermarking, jia2025multiembedding, dong2026residual,
wang2026resguard, nemecek2026contradictions, li2026reissuance, kazempourradi2026softbindings,
abud2026warp. Bibliographic data only (not read): mintzer1999more, mascher2006multiple.
Not added: PAI (not relevant), WAVES, Sencar 2007, the patent.
