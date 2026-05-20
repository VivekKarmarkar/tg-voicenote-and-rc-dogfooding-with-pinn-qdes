# PINN-QDE Paper — Introduction Section Draft

This is a Step 2 synthesis of `introduction_verbatim.md`, `introduction_claude.md`, `introduction_reference_notes_*` and `introduction_reflections_*` into early-stage prose. The arc is Vivek's own; the wording is half his (verbatim signature phrases preserved), half my structuring.

---

## 1. The AI Wave Reaches Physics

AI is a broad umbrella — chess engines and Deep Blue, AlexNet on cats and dogs, the post-2012 neural network explosion, generative AI as the latest wave. But the thing that made AI ubiquitous and household was specifically the browser-based large-language-model chat interface: ChatGPT, browser Claude. There is no putting the genie back in the bottle. The LLM-driven mainstreaming is bleeding into the rest of the AI space and into science itself, evidenced at the highest level by the 2024 Nobel Prizes: Physics to Hopfield and Hinton for the foundational work that enabled the current wave — Hopfield networks, Boltzmann machines, the architectures that made backpropagation tractable — and Chemistry to Hassabis and Jumper at DeepMind for AlphaFold, which closed out a decades-old problem in protein structure prediction.

## 2. The AI–Physics Bidirectional Channel

When AI influences a field as old and as hard as physics, the channel runs both ways.

**Physics → AI.** AI systems don't run on thin air; the substrate matters. The fundamental arithmetic of a neural network — convolutions, matrix products — can be carried out optically, and far more energy-efficiently than in CMOS silicon. Optical neural networks, optical computing, and quantum computing are all emerging precisely because physics has something tangible to offer the hardware layer of AI.

**AI → Physics.** Physics is observing and understanding nature. You probe nature with experimental apparatus, you collect data, you make sense of the data either by discovering a fundamental law or by solving a downstream engineering problem. AI enters at both ends of that chain: at the experimental design level, and at the data analysis level.

At the experimental level, AI can narrow what would otherwise be a brutally large configuration space. Mario Krenn's group at the Max Planck Institute for the Science of Light (his PhD advisor Anton Zeilinger is a Nobel laureate) has used AI to discover quantum-optics experimental setups that human researchers would not have intuited — including entanglement configurations of high complexity. AI can also become part of the experiment itself: Roark Horstmeyer's physical neural networks integrate a CNN classifier with a programmable illumination pattern, simultaneously co-optimizing the physical illumination (which is, literally, the first "layer" of the network) and the trained CNN weights. The AI is not analyzing the experiment from outside; it *is* the experiment.

Once data is collected, the goals split three ways: understand the data for its own sake, discover a fundamental physical law, or solve a problem (which bleeds into engineering). All three are increasingly being attacked with AI.

## 3. Scientific Machine Learning

Many physical and engineering systems are governed by differential equations. The broad field of AI that deals with understanding, solving, and discovering differential equations from data, focused on physics and engineering applications, is *scientific machine learning* (SciML). SciML decomposes into three rough pillars:

- **Equation discovery.** Max Tegmark's AI Feynman, Miles Cranmer's PySR (symbolic regression), Steve Brunton's SINDy (sparse identification of nonlinear dynamics — sparse regression over a hand-chosen library of functions). All target the inverse problem of going from observed dynamics back to a closed-form governing law.
- **Operator learning.** Neural operators (Anandkumar, Mishra, Karniadakis), neural inverse operators for inverse problems, and the Universal Differential Equations (UDE) framework of Rackauckas. The move is to learn the *solution operator* of a PDE family rather than a single instance.
- **Physics-informed neural networks (PINNs).** The Karniadakis paper that opened this line has around 20,000 citations. The umbrella name "PINNs" covers everything that combines neural networks with physics, but the load-bearing case is the narrow one: a neural network parameterizes the solution to a differential equation, and the equation's residual is folded directly into the loss function.

The critical context: scientific research groups are not Google or Amazon. They do not have, and structurally cannot get, internet-scale data. But they *do* have mechanistic understanding of their system — the differential equation. PINNs and SciML methods in general bridge limited data with underlying physics. That fusion is the entire point.

## 4. PINNs and Neural Fields — Viable, Not Just Popular

PINNs are not just citation-popular; they are solving real problems where conventional methods fail. The same idea also travels under the name "neural fields," especially in the imaging and computer vision community.

Katie Bouman's group at Caltech has used neural fields for astrophysical imaging and gravitational lensing — reconstructing the dark matter distribution that produced a single observed lensing pattern. This is a regime where data is fundamentally scarce: you cannot just go collect more lensing events from a different angle. The inverse problem is genuinely ill-posed by any classical standard, and neural-field methods solve it. In super-resolution microscopy, similar ideas have been used to combine a U-Net backbone with the underlying convolutional physics of the imaging system, resolving structures that conventional inversion cannot reach.

Beyond the specific wins, the structural advantages over legacy methods matter:

- Mesh-free: no domain-specific grid generation.
- Robust to limited data: physics fills the gap.
- Fast to run, easy to scale, easy to adopt.
- Built on modern, battle-tested differentiable frameworks (PyTorch, JAX), not legacy Fortran solvers.

The result is that PINNs are not just trendy — they are a genuinely viable foundation to dig deeper into.

## 5. The Gap — From Real to Complex to Quaternionic

Many of the differential equations that arise in physics and electrical engineering are most naturally expressed over the complex numbers. In electromagnetism, the family of solutions to Maxwell's equations is built from complex exponentials, and the algebra is cleaner in a complex-valued framework — that is convenience. In quantum mechanics, the wave function is intrinsically complex-valued — that is not convenience. The 2021 Nature paper by Renou et al. closes this question decisively: a Bell-like experiment falsifies any real-valued reformulation of quantum theory. It is not that complex numbers are an artifact of how QM was built; there is no real-valued alternative that reproduces experiment. Complex numbers are *load-bearing* in QM, in the strongest sense.

Yet most PINN implementations, even when applied to a complex-valued physical problem, place a real-valued neural network in the latent space and let its real-valued outputs *represent* a complex quantity at the boundary. That is a workable starting point, but it is patching.

The same algebraic story repeats one dimension up. Rotation is fundamental in physics and engineering, and the operations that capture rotation — the dot product and the cross product — are introduced early in any physics or engineering curriculum as separate definitions, almost by decree. But the dot and cross products are intrinsically tied to rotation, and rotation is most naturally described by *four* numbers: a unit axis (three numbers) and an angle about it (one number). That four-number object is a quaternion — a hypercomplex number — and the dot/cross product structure falls out of quaternionic multiplication. Quaternions are to rotation what complex numbers are to QM and EM. They are the natural algebra.

Yet quaternionic neural systems are an underexplored niche, sitting at the intersection of three communities that do not really talk:

- The motion-tracking / IMU community uses quaternions everywhere but thinks in Kalman filters and signal processing. For them, neural networks are a black-box classifier or fast predictor, not a fundamental object of study.
- The PINN community has explored complex-valued PINNs (CompleX-PINN, complex-valued PINNs for neutrino oscillations) but has not seriously taken on quaternionic differential equations.
- The applied-math community has developed quaternion neural networks (QNNs — Zhu et al. 2019, Parcollet et al., Zhang et al. Quaternion Product Units) and the theory of linear quaternionic differential equations (Kou et al. 2018), but has not connected to PINNs or rotational dynamics as a target problem.

The closest existing work is TE-PINN (2024), a transformer-enhanced PINN that uses quaternion kinematics for orientation estimation — but this is sensor fusion, not solving quaternionic differential equations. No published work uses PINNs to *solve QDEs* in the Karniadakis sense.

## 6. Contribution

This paper extends PINNs in the Karniadakis sense to quaternionic differential equations, focused on rigid body rotational mechanics, validated on real-world industry-grade IMU sensor data from Movella.

There are two design options for "a PINN over a quaternionic algebra":

a. **Intrinsic.** Build a neural network whose latent space is quaternion-valued throughout: quaternionic weights, quaternionic activations, quaternionic backpropagation, derivatives defined in the quaternionic sense. This is the deep direction — and at this point, the answers to "how do you define a derivative here" and "how do you backpropagate" are not in any battle-tested library.
b. **Pragmatic.** Use a real-valued neural network out of a battle-tested library (here, JAX), customize the output to represent a unit quaternion, and hand-code the quaternionic algebra so it composes cleanly with the framework's automatic differentiation. The latent space stays real; the symbolic quaternionic structure lives in the loss function and the operators.

We take approach (b). The contribution is the actual JAX customization: building a unit-quaternion-valued output head, building the quaternion algebra (Hamilton product, conjugate, inverse, log map for geodesic distance) such that every operation is fully compatible with the autodiff engine, and stitching it all together into a PINN that solves the rigid-body rotational equations of motion.

The headline empirical findings:

1. **Vanilla PINNs fail.** A naive out-of-the-box PINN, even with the quaternion machinery wired in, cannot fit the IMU rotational data. The failure mode is spectral bias — the network preferentially fits the low-frequency content and is structurally bad at the higher-frequency rotational components present in the data.
2. **The diagnostic.** Wavelet scalograms turn the failure into a visible, interpretable picture of where in the time-frequency plane the network is and is not tracking the signal. They are a practical smell-and-taste tool for diagnosing spectral bias on real time series.
3. **The fix.** Fourier features at the input, combined with a quaternionic-distance-aware initial-condition loss term, restore performance. The full custom stack — Fourier features + JAX-compatible quaternion operations + IC loss term that minimizes geodesic distance on the unit-quaternion sphere — extends the PINN cleanly to quaternionic differential equations.

The takeaway is sharper than "we made PINNs work for one more equation class." Pushing PINNs to a fundamentally interesting algebraic regime *reveals* what they can and cannot do out of the box, and the diagnostics that surface in the failure mode (spectral bias, visible in wavelet scalograms) are general lessons about PINN training that the rotational case happens to make especially legible.

## 7. Framing and the Honest Caveat

Extending PINNs to QDEs is an instructive exercise for the algorithmic development of PINNs precisely because the underlying object — the quaternion — sits so close to the elementary content of every physics and engineering education, and yet has not been treated as a first-class object in the SciML stack. Pulling it in forces hand-coded custom operations that must compose cleanly with autodiff, which exposes design choices that are invisible when everything is real-valued.

The honest caveat: our approach is, at a fundamental level, *patching*. A real-valued latent space dressed up to output unit quaternions is not the same thing as a network whose internal representations live in $\mathbb{H}$. The intrinsic quaternion-valued neural network — with quaternionic backpropagation, derivatives in the quaternionic sense, and the move *away* from the comfortable JAX/PyTorch stack — is the deeper direction. We frame that as future work and as the place where the real gravitas of this niche lives. This paper establishes that the pragmatic extension works, on real data, with the failure modes and fixes made explicit; the intrinsic direction is the natural next step.

---

## Source map (for downstream LaTeX step)

- Sec. 1 — `introduction_verbatim.md` messages 2252, 2254.
- Sec. 2 — `introduction_verbatim.md` messages 2254, 2256.
- Sec. 3 — `introduction_verbatim.md` message 2258.
- Sec. 4 — `introduction_verbatim.md` message 2260.
- Sec. 5 — `introduction_verbatim.md` message 2262 + `introduction_reference_notes_*.md` (Renou et al. 2021 Nature) + `literature_review/QNNs_agent_first_pass_lit_review.md` (TE-PINN, CompleX-PINN, QNNs, Kou et al.).
- Sec. 6 — `introduction_verbatim.md` messages 2262, 2264.
- Sec. 7 — `introduction_reflections_verbatim.md` message 2367 (the "patching vs intrinsic" reflection).
