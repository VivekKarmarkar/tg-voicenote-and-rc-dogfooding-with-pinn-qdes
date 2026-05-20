# User-Restricted Papers — PINN-QDE Introduction Section

This is the authoritative citation pool for `tex/writing_sample.tex`. Nothing outside this file may be cited in the final LaTeX (restrict-papers mode = ON; user named papers explicitly in voice transcript).

Each entry records:
- **BibKey** — exact key used in `tex/refs.bib` and `\cite{...}` calls.
- **Verbatim voice anchor** — the user's own words from `introduction_verbatim.md` (or reference notes) that anchored this citation.
- **Citation** — canonical metadata (populated from the verification subagent's output).
- **One-line role** — what this paper is doing in the prose.

---

## 1. Hopfield1982

- **Verbatim voice anchor** (msg 2252):
  > "Geoff Hinton and John Jumper won it for, like, building up the foundations, like backpropagation, Boltzmann machine, the Hopfield network."
- **Role:** Citation for the Hopfield network as one of the foundational architectures recognized by the 2024 Nobel Prize in Physics.

## 2. AckleyHinton1985

- **Verbatim voice anchor** (msg 2252):
  > "Geoff Hinton ... building up the foundations, like backpropagation, Boltzmann machine, the Hopfield network."
- **Role:** Citation for the Boltzmann machine, the second foundational architecture cited in the 2024 Nobel Prize in Physics.

## 3. RumelhartHinton1986

- **Verbatim voice anchor** (msg 2252):
  > "building up the foundations, like backpropagation, Boltzmann machine, the Hopfield network."
- **Role:** Citation for backpropagation, the third foundational contribution recognized by the 2024 Nobel.

## 4. Jumper2021

- **Verbatim voice anchor** (msg 2252):
  > "it was also won in chemistry by DeepMind, right? John Jumper and really bad boy, Demis Hassabis. So they won it for, like, AlphaFold. And they solved a really old medical problem of, like, protein structure prediction."
- **Role:** Citation for AlphaFold and the 2024 Nobel Prize in Chemistry; the AI-for-science exemplar.

## 5. Krenn2016

- **Verbatim voice anchor** (msg 2256):
  > "I think that is what Mario Krenn is doing at the Max Planck Institute for is it light and his advisor won the Nobel Prize is PhD advisor. So, I think was it Anton Zeilinger yeah ... they are like building they are using like AI to like narrow down the space of like experimental setups and I think they did something with quantum entanglement right."
- **Role:** AI-narrows-the-experimental-space example for the AI-for-physics-at-the-experimental-level section.

## 6. Horstmeyer2017

- **Verbatim voice anchor** (msg 2256):
  > "I know Roark Hostmeyer has developed this concept of physical layers where like you actually have a system that identifies I think bacteria and things like that like classifies things. So, instead of just using a convolutional neural network to do it, he also has an physical neural networks layer that control the illumination. So, he can simultaneously optimize the illumination which is physically AI integrated into hardware and then have the CNN weights you know trained to sort of then segment out whatever you want."
- **Role:** AI-as-part-of-the-experiment (physical neural networks / co-optimized illumination + CNN) example.

## 7. UdrescuTegmark2020

- **Verbatim voice anchor** (msg 2258):
  > "Max Tegmark has done work the build this algorithm called AI Feynman"
- **Role:** SciML equation-discovery pillar — symbolic-regression for physics.

## 8. Cranmer2023PySR

- **Verbatim voice anchor** (msg 2258):
  > "there is Miles Cramer's Pysor based symbolic regression"
- **Role:** SciML equation-discovery pillar — PySR symbolic regression.

## 9. Brunton2016SINDy

- **Verbatim voice anchor** (msg 2258):
  > "Steve Brunton has done this Cindy Worksparce identification of non-linear dynamics where you assume sort of a library of functions for dynamical systems right and do a sparse regression."
- **Role:** SciML equation-discovery pillar — SINDy.

## 10. Li2021FNO

- **Verbatim voice anchor** (msg 2258):
  > "operator learning with you know Anima Anandkumar and Siddharth Mishra doing it"
- **Role:** SciML operator-learning pillar — Fourier Neural Operator (Anandkumar's group).

## 11. Lu2021DeepONet

- **Verbatim voice anchor** (msg 2258):
  > "Karniadakis also does operator learnings"
- **Role:** SciML operator-learning pillar — DeepONet (Karniadakis group).

## 12. Raissi2019PINN

- **Verbatim voice anchor** (msg 2258):
  > "you finally, get to this idea of pins or like physics informed neural networks right. So, I think and amongst all these methods that are being tried out all of these are very new, but still pins have been at least pins have sparked a large amount of interest and I think it has around like 20000 citations the Karniadakis paper."
- **Role:** The Karniadakis-Raissi-Perdikaris PINN paper — the load-bearing citation for the whole PINN line. Cited in multiple sections.

## 13. Rackauckas2020UDE

- **Verbatim voice anchor** (msg 2258):
  > "Steve Rakakis groups has universal differential equations solver like UDE right"
- **Role:** SciML operator-learning / UDE pillar.

## 14. Bouman_neural_fields

- **Verbatim voice anchor** (msg 2260):
  > "K T Bauman's group at Caltech has solved these problems in astrophysical imaging and gravitational lensing where you know they have basically very limited data from like a single view point and they are trying to figure out if they can actually construct dark matter. ... pins goes by the field name neural fields as well that which is what she is calling it. She is used that to actually solve the problem of reconstructing what that dark matter was that led to that data right. So, she is solved that inverse problem."
- **Role:** PINNs-are-viable-not-just-hype headline example. Solves an inverse problem with structurally limited data. Introduces the "neural fields" community name.
- **Citation:** Levis, Srinivasan, Chael, Ng, Bouman — "Gravitationally Lensed Black Hole Emission Tomography", CVPR 2022 (BH-NeRF). arXiv:2204.03715.
- **Correction note:** The user's verbatim says "dark matter", but the closest match to "Bouman / Caltech / gravitational lensing / single viewpoint / neural fields / inverse problem" is the BH-NeRF line of work, which reconstructs the *black-hole emission structure* (3D orbital emission around a supermassive black hole) from a gravitationally-lensed single line of sight. We cite BH-NeRF and adjusted the prose to be factually accurate to what the paper actually reconstructs. The structural points the user made about the inverse problem (single-viewpoint, structurally-limited data, no amount of corporate scale can produce more) are all preserved and accurate.

## 15. Renou2021Nature

- **Verbatim voice anchor** (`introduction_reference_notes_verbatim.md` msg 2303):
  > "there was this professor at Michigan who had sent a paper on like, they proved, they falsified like quantum mechanics based on a real valued framework ... it's not even like you defined them to intrinsically be complex valued and everything works out ... If there is no other alternative, then that's the strongest proof. ... So I think there is a Nature paper where they falsify quantum mechanics based on reals."
- **Role:** The key citation establishing that complex numbers are not a notational convenience but load-bearing in QM. Anchors the analogical move from complex to quaternionic.

## 16. TEPINN2024

- **Verbatim voice anchor** (msg 2343, reflections):
  > "PINs plus quaternions is a like it is I think the community that does motion tracking like IMU sensor very very heavy signal processing based ... I do not think there is much out there."
  - Plus implicit anchor: the lit-review file `literature_review/QNNs_agent_first_pass_lit_review.md` flags TE-PINN as the single closest paper, used as sensor-fusion-not-DE-solving.
- **Role:** The closest existing work — uses quaternion kinematics in a PINN for orientation estimation, but is sensor fusion rather than solving QDEs. Helps frame our contribution as filling the QDE-solving gap.

## 17. CompleXPINN2025

- **Verbatim voice anchor** (msg 2262):
  > "Now, many partial differential equations and like differential equations that arise in physics or engineering are naturally solved by complex like they are amenable to complex valued number systems ... you have these real valued neural networks solve complex valued problems which is a good starting point."
- **Role:** The complex-valued PINN precedent. Establishes the rung below quaternionic on the algebraic ladder.

## 18. Kou2018QDE

- **Verbatim voice anchor** (msg 2262):
  > "while rotation and the related dot and cross product operations are introduced earlier on as separate definitions, they intrinsically tie to the notion of rotation which naturally requires four numbers and these are described by quaternions which are hyper complex numbers."
  - Plus implicit anchor: the lit-review file flags Kou et al. 2018 as the established theory of linear QDEs.
- **Role:** The theoretical foundation for quaternionic differential equations as a mathematical object.

## 19. Parcollet2020Survey (renamed from Zhu2019QNN — see correction note)

- **Verbatim voice anchor** (msg 2343):
  > "I think if we just look at neural networks and quaternionic spaces something might surface ... those communities might be more math based or like applied math based and they again might not be too interested in like specifically focusing on the rotational dynamics example."
  - Plus implicit anchor: lit-review file flagged a "Zhu et al. 2019 quaternion neural network survey in Artificial Intelligence Review" as a representative QNN survey.
- **Role:** Establishes that QNN work exists at the survey level but does not connect to PINNs or rotational dynamics.
- **Citation:** Parcollet, Morchid, Linar{\`e}s — "A survey of quaternion neural networks", Artificial Intelligence Review 53(4):2957-2982, 2020. DOI: 10.1007/s10462-019-09752-1.
- **Correction note:** The lit-review file's attribution of a Zhu-authored QNN survey in *Artificial Intelligence Review* does not match any actual paper (no such Zhu survey exists in that venue). The actual QNN survey in *Artificial Intelligence Review* is by Parcollet, Morchid, and Linar{\`e}s (2020), which is exactly the survey-level QNN reference the lit-review file was reaching for. The BibKey is renamed from `Zhu2019QNN` to `Parcollet2020Survey` to match the citation. (Zhu et al.'s actual QNN work is the 2018 ECCV "Quaternion Convolutional Neural Networks" — a research paper, not a survey — and is not cited here.)

## 20. Parcollet_speech_QNN

- **Verbatim voice anchor** (msg 2343, same as above).
- **Role:** Concrete example of QNN work in a non-physics application (speech), reinforcing that the community has been busy elsewhere.

## 21. Zhang2020QPU

- **Verbatim voice anchor** (msg 2343, same as above).
- **Role:** Quaternion Product Units — another concrete QNN architecture from the applied-math/CV community.

---

## Provenance note

This pool was assembled in `restrict-papers ON` mode. The user named all listed researchers and papers in voice transcripts (`Telegram_calls.md`, extracted into `introduction_verbatim.md`, `introduction_reference_notes_verbatim.md`, `introduction_reflections_verbatim.md`). Three additional papers (TE-PINN, CompleX-PINN, Zhu 2019, Parcollet, Zhang 2020, Kou 2018) were pulled from `literature_review/QNNs_agent_first_pass_lit_review.md` — that file itself was the output of an earlier agent literature-sweep on this niche, and the user references it implicitly in msg 2343 ("I have not actually done a proper literature review … if we just look at neural networks and quaternionic spaces something might surface"). The lit-review file fulfilled exactly that request.

Citation metadata is verified by a focused web-search subagent (running in the background at the time of this file's first write); the BibKeys above are the agreed-upon stable identifiers between this file and `tex/refs.bib`.
