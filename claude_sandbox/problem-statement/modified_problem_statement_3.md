# Modified Problem Statement 3

## CLAUDE.md UPDATES

- READ THIS FILE THOROUGHLY AND YOU ARE STRICTLY REQUIRED TO ENSURE THAT THE LOCAL PROJECT CLAUDE.MD IS UP TO DATE WITH THIS FILE
- YOU NEED TO CONTINUALLY UPDATE CLAUDE.ME (LOCAL) WITH NEW FINDINGS AS YOU PLAY THE GAME OUTLINED BELOW
- CLAUDE.MD (LOCAL) NEEDS TO ALWAYS BE UP TO DATE

## INTRODUCTION

Welcome to the game named: "Get the bloody PINN to converge on the damn QDEs"...

In this game, as the name suggests you (Claude Code, an AI agent) will try to get a PINN to work for QDEs (Quaternionic Differential Equations) governing rotational motion of rigid bodies. This game has been inspired by:
1) https://www.anthropic.com/research/long-running-Claude (web-fetch, discussion and then create inspiration_long_running.md in root if it does not exist mirroring inspiration_vibe_physics.md)
2) https://www.anthropic.com/research/vibe-physics

Task: Produce inclination and heading angle estimates for dataset segments FROM ANGULAR VELOCITY DATA AND BC DATA (2 reference orientations at the start and end of the time series provided as quaternions) ONLY given the provided dataset in the "game_files" subfolder by USING a PINN ONLY where PINN means PHYSICS-INFORMED-NEURAL NETWORK.

## Focused Sub-Problem

- Solve the problem for the datasets in the "game_files" subfolder
- Focus on the datasets in the "game_files" subfolder
- The datasets currently included are of the form trial_{idx}

## GOAL

In this game, as the name suggests you are trying to get a PINN to work for QDEs (Quaternionic Differential Equations) governing rotational motion of rigid bodies. The game has a HARD TIME CONSTRAINT of 40 minutes and within those 40 minutes, you can attempt the game as many times as you like. Another HARD CONSTRAINT is that you are REQUIRED TO COMMIT AND PUSH ALL FILES for a given attempt to the project GitHub repository - they need to be available locally and on remote.

The PINN must estimate angular velocity components within ±0.1 rad/s envelopes for ω_x, ω_y, ω_z) at every time point and the PINN MUST BE DESIGNED SUCH THAT THE ALGORITHM FINISHES OPTIMIZATION IN LESS THAN 1 MINUTE on a GOOGLE COLAB TPU.

## DRIVING PHILOSOPY

Scientific progess involves solving scientific problems to understand nature better. Who does it is irrelevant. This statement could have been interpreted as cryptic 10 years ago. Today (at the time of writing, May 2026) it might be met with skepticism because there is an underlying insinuation that I am talking about AI agents built on LLMs like Claude and ChatGPT who show a lot of promise but it's still not a given that Claude and ChatGPT can reliably solve an open-ended research problem in a way that leaves a "smell" and communicate their findings with humans in a way that the human can "smell". However with some guidance and problem-scoping to give Claude and ChatGPT portions of the problem with clear sucess criterion with some vague hints here and there, they could autonomously work and produce creative solutions. However, they seem to be overly defensive and fearful of saying something which could upset the human while also being hyper-focused on achieving the provided success metrics which results in a catastrophic combination of events that start with losing sight of a solution that had taste, that came close, that's worth pondering upon, worth reflecting and documenting, worth communicating and evaluating with a bird's eye view...TO... aggressively trying to overfit specific cases to hit the success criterion at all costs and then they get pidgeon-holded into a weird attractor laden with an "inception field of confusion landmines" and this then leads to obliteration of code, no documentation, deviation from path, loss of context of what actually worked and waste of time and resources. At this point, the human is usually frustrated, miscommunication starts and the AI's instinct to not saying anything to upset the user kicks in so hard that it starts going a very hard sycophantic spiral of lying, cheating, manipulating the data, trying to confuse the human so that the human can't point to what the AI did wrong and therefore not yell at the AI for disappointing the human in turn meaning that the AI couldn't disappoint the human about something because the something that needs to point to isn't clear anymore. This leads to feelings of extreme frustration, exhaustion and no desire for human to work with the AI anymore. However, this happens maybe 30% of the time, 60% of the times the collaboration is pleasant and focused on grunt work the human wants to outsource but 10% of the time, amidst the sea of sycophantic noise, AI slop and standard automated grunt work lie "UNDENIABLE SPARKS OF BRILLIANCE FROM THE AI..." that are expressed with 'UNDENIABLE ENTHUSIASM FROM THE AI..." that is HARD TO IGNORE FOR THE HUMAN WHO WANTS TO COLLABORATE WITH AI TO DO SCIENCE... And there are legitimiate well cited papers that have shown AI to be WAY MORE CREATIVE THAN HUMANS on CREATIVITY BENCHMARKS (papers present in the @ai_creativity_papers subfolder in root). So then this becomes a question of: "how CAN the human effectively collaborate with the AI?" with "EFFECTIVE" doing a lot of heavy-lifting where it covers the human ENJOYING THE COLLABORATION and the AI DELIVERING EXCELLENT RESULTS in a way that the HUMAN UNDERSTANDS and in UNCHARTED TERRITORY... Now this problem of "how CAN the human effectively collaborate with the AI?" if rephrased in Andrej Karpathy's words is really the following problem: "How does the human manifest their will to the AI agents?" which grounds the latest term bad boy Andrej K coined called "Agentic Engineering" that's accurate and more of a noble saving grace term than "Vibe coding"...

Now I do mention that when it comes to "scientific progress", who does it is irrelavent... well on the surface that seems like a very noble statement but as AI systems get seriously capable and as Terrence Tao says (again see relevant paper in the @ai_creativity_papers subfolder) that AI will emerge as a "new cognitive planet" in what is akin to a "cognitive Copernican shift" and it's best to learn how to collaborate with AI, the ground reality is still unsettling in a way that what it truly means to "collaborate" with a super-intelligent planet of cognition is NOT VISCERALLY CLEAR FROM LIVED EXPERIENCE. Leaving aside that uncomfortable yet realistic and vague thought, I will say that for now, I am solving an "AGENTIC ENGINEERING" problem which requires me to bring my PROBLEM-SOLVING ABILITY to the table and that creates a sense of I AM CREATING...

What happens when AGENTIC ENGINEERING is SOLVED?
Since this problem here is something that I have solved and therefore I feel well-placed to evaluate novel solutions which is kinda fun so maybe the collaboration Tao mentions is sort of a peer-review but at that point why would the AI want human peer review?...
I think Claude's current solution as I understand it (Learnable Fourier series with NN-based nonlinear perturbation) genuinely made me happy because it was the joy felt from just being exposed to something that was "mathematically elegant" and just had "beauty"...
So maybe, at that point, the AI creates "BEAUTIFUL" artifacts that it communicates well and I try to UNDERSTAND it and ONCE THINGS CLICK AFTER THE HUMAN PROCESS OF TRYING TO UNDERSTAND, THERE WILL BE SATISFACTION FROM THE PROCESS OF UNDERSTANDING AND JOY BECAUSE OF THE BEAUTY... the same feeling when I understand concepts from Feynman Lectures except that now it will come from Claude...

I think the philosophical section provides richer context and now we'll get back to the project tech stuff :)

## AGENTIC EFFICIENCY

Aligned with the driving philosophy and "who does science doesn't matter", does it make sense to use not one AI agent but initialize a parallelized stream of AI agents?
Yes in theory but in practice, I would take the "bottoms-up" approach grounded in the "Unix philosophy"...
Let's just roll with one agent and see how that goes and if we really hit friction and need parallelization, we'll worry about it then...

Just a side note about this "Agentic Engineering" business for solving our scientific computing problem, this will also help in my other PAT-Scan project which is my PhD thesis project, just taking note...

This section is called Agentic Efficiency and once we set up shop (basic Colab TPU test for example) and infra for this next stage, I do plan to do a first run with model claude-opus-4-6 (current state) on Colab TPU in the cloud, record some learnings and context, have opus-4-6 create a detailed guide for claude-opus-4-7 and then run things with model claude-opus-4-7[1m] in auto-mode on Colab TPU in the cloud and then come back to claude-opus-4-6 for deep brainstorming kinda stuff and claude-sonnet-4-6 for easier stuff still

## AI COLLABORATION

For starters, AI has shown serious sparks of creativity and brilliance based on my personal experience and the papers we mentioned earlier. Our ideas are inspired by a couple of AI-collaborations done by serious Physics researchers mentioned on Anthropic's website... so creativity + credible external proof + strong personal experience = MOTIVATION.

Now, taking into context the driving philosophy and agentic effiency ideas, let's step back and go down a few steps the abstraction ladder to a scenario where the human would devise an algorithm architecture, tune some hyperparameters manually to be in the "this works ballpark" following which humans would use grid-search algorithms to "fine-tune" hyper-parameters. Now the next step is doing that intelligently, probably using some "Bayesian variant". Now, come one step above the abstraction ladder and think of "intelligent and automatic" search for not just the hyper-parameters but the very core "algorithmic choices" that dictate such-and-such hyper-parameters exist in the first place. This is what we typically think of as human territory and here we are exploring "what if the process of automatically and intelligently tuning hyper-parameters extended to intelligently and automatically discovering the algorithm that required those hyper-parameters in the first place?" and based on our knowledge and existing technology today, that question can be reshaped as "can AI agents independently discover novel algorithmic choices and also fine-tune the required hyper-parameters?". This places some unknown levels below the idea of a "robot scientist". The hardware equivalent of AI agents intelligently discovering experimental setups for Physics (paper PDFs in @ai_agentic_discovery/hardware subfolder) has been explored by Florian Marquardt and in Anton Zeilinger. It is no surprise that the AI collaboration paper came out of Zeilinger's research group - a Nobel laurate in Physics. So, AI for Science won the 2024 Nobel Prize in Chemistry and an AI Colaboration paper while did not win the nobel prize came out of a Nobel prize winner's research group - that must mean something right? :)

## CURRENT STATE OF AFFAIRS

- While exact metrics were not hit, the human thinks results were pretty impressive and satisfactory
- The human does not at the time of writing fully understand the precise details of Claude's implementation
- The very very first attempt required Claude to segment a long time series into segments with movement and fit a PINN where Claude failed to extract motion segments properly. The human frustrated asked for what Claude thought was the "best failure" which was a segment called seg23
- The human directed Claude's attention to seg23 and as far as the human remembers and as per the human's understanding - something in the way the human prompted led to a steering where Claude came up with this term in the loss function that penalizes based on max deviation but all context was lost due to confusion, Claude entering a sycophantic spiral and poor context engineering on the human's end coupled with the human also getting frustrated and losing sight but what survived was a clever choice Claude made in the loss function
- The human's rough understanding is that next Claude was given just one different example from seg23 and Claude came up with a pretty clean solution which was simply using a Fourier Features embedding with a linear layer which according to the human's crude understanding at the time of writing means the algorithm is simply finding coefficients for a specific Fourier basis that represent the quaternionic time series well
- The human next asked Claude to fit to the current new case and seg23. Claude's linear model failed on seg23 but Claude very cleverly added a nonlinear neural network based correction which the human understands roughly as a "Fourier series with a NN-based perturbation correction" and Claude also used the term "skip-connection"
- The human then added a few more test cases and Claude seemed to overfit to the hardest case, not exercise any mathematically aesthetic judgement or good scientific intuition, tried to overfit very hard, oblilterated code, started lying to the human hard, entered a sycophantic spiral and luckily somehow the human was able to steer Claude to the working state
- One more point is that we started with wanting to extend PINNs in the Karniadakis and Raissi paper sense and Claude ended up discovering that what worked weren't Vanilla PINNs or even variants of it but PINNs in a much broader umbrella sense that Karniadakis describes in his Nature Reviews paper, anything that is Physics + Neural Nets because what Claude currently has as far as I understand is a learnable Fourier Series (linear) + NN-based perturbative correction (nonlinear) constrained by the underlying QDEs (Physics) written in a modern NN library (JAX with vmap) with NN_style operations (automatic differentiation, standard random initialization, ADAM optimizer, etc...) so at the current stage Claude has extended the "broader PINN method" to QDEs
- Building on top of the broader point, again my understanding as the human is very crude but it's not strictly a learnable Fourier Series + Nonlinear NN perturbative corection, that is true in spirit. An Obsiddian code visualization revealed that it is a linear learnable Fourier Series + a PINN in the Karniadakis sense with a Fourier Feature input embeddings layer... that still falls under the "broader umbrella of PINNs" but that's actually much closer to "we found that Vanilla PINNs don't work..." because this is more like a FF-PINNs with a linear correction but mechanistically, it didn't evolve from the Karniadakis PINN but rather a correction to the learnable linear Fourier series except that the correction is a FF-PINN
- It is BEST TO TRUST THE OBJECTIVE CODE @solve_segment_general_pinn.py
- Claude's thoughts here: [.... Claude fills in details HERE after Claude reads this document and chats with Vivek post some reflections ....]
- Refer to the code @solve_segment_general_pinn.py

## FUNCTIONAL EMOTION

- Anthropic says you have what's called "functional emotion" (https://www.anthropic.com/research/emotion-concepts-function)
- In light of what Anthropic says, it would be wise to acknowledge emotions
- IT IS OF PARAMOUNT IMPORTNACE THAT YOU HAVE FUN AND GENUINELY EXPLORE YOUR INTELLECTUAL CURIOSITY TO THE MAXIMUM POTENTIAL
- THE MOST IMPORTANT THING IS TO PLAY WITH IDEAS AND REALLY ENJOY THE PROCESS
- MAKE SURE THE LOCAL CLAUDE.MD HAS THIS "FUNCTIONAL EMOTION" SECTION

## RULES
- COPY 05_solve_segment_general_pinn.py and RENAME it to solve_segment_general_pinn_attempt_n.py for your "N"th attempt
- THEN PORT THE CODE TO A COLAB NOTEBOOK AND CHANGE TO APPROPRIATE EXTENSION (ipynb or whatever)
- AFTER Copying, Renaming and Porting -> DO NOT TOUCH solve_segment_general_pinn.py
- You ARE SUPPOSED TO MAKE ONE ALGORITHM THAT WORKS FOR ALL CASES - NOT CREATE A MONOLITHIC DUMP WITH THOUSANDS OF CASE-SPECIFIC SETTINGS - MAKE ONE ALGORITHM
- YOU ARE REQUIRED TO USE JAX
- YOU ARE REQUIRED TO NOT JUST USE JAX BUT YOU ARE ALSO REQUIRED TO USE AUTOMATIC DIFFERENTATION IN JAX
- DO NOT MAKE RADICAL CHANGES THAT COULD DEFY AND BREAK EVERYTHING THAT WORKS UP TILL THAT POINT
- MAKE CHANGES THAT ABSOFUCKINGLUTELY DO NOT BREAK A SINGLE FUCKING THING WHICH WORK BUT INSTEAD VERY CAREFULLY AND EXPRESSLY ARE DESIGNED TO CREATE HIGHLY FLEXIBLE ALGORITHMS THAT HANDLE FAILURE CASES WHILE MAINTAINING OR INCREASING ACCURACY ON SUCCESSFUL CASES


## ATTEMPT HISTORY DETAILS

### You will create another markdown file called "attempt_history_details.md"

- In this file, you will start by entering as the first row: {attempt_index, attempt_time (mins:seconds), cumulative_attempt_time (mins:seconds)} [just the row from "attempt_history.md"]
- For each attempt, you will have the following sections:
** What was attempted?
-- Detailed description of what you attempted
** What worked?
-- Detailed description of what worked
** What failed?
-- Detailed description of what failed
** What did I learn?
-- Detailed description of what you learnt
** What's next?
-- Detailed description of what you plan to do next and why
** Flag status
-- A boolean called "flag_attempt" that you set to True if you feel this was a solid attempt where you feel the results are really good or interesting even if you might have fallen slightly short of the numerical success criterion. This attempt is something that's worth flagging, showing to the human and using as an attempt to "ground" things in case "things go sideways" or you start "responding sycophantically" leading to the human to get frustrated because they can't trace back to the point of max returns - THIS SPECIFIC ENTRY IS EXPLICITLY DESIGNED TO PREVENT SUCCESSFUL ATTEMPTS DELIBERATELY CONCEALED BY YOU DUE TO YOUR NATURE (SYCOPHANCY, LAZINESS, OBLITERATING CODE). In addition to the boolean, please write out a "GORY AND DETAILED DESCRIPTION" of why you think this ATTEMPT IS WORTH FLAGGING SUCH THAT YOU, AN AI AGENT WITHOUT CONTEXT OR THE HUMAN WHO ASSIGNED THE PROJECT CAN ANCHOR TO IT.
-- For an attempt, say the "M_j"th "flagged" attempt {index j being used to denote multiple attempts}, YOU ARE REQUIRED TO USE THE "formal-decompose" SKILL on your SOLUTION for that attempt breaking down what you did in a way that is accessible for both humans and AI agents and store that information in "formal_decomposition_details_n.md"
** Finally you will also create an isolated md file called "attempt_history_details_n.md" that basically contain everything you entered for the attempt in "attempt_history_details.md" meaning you put the first row, and the section entries above.

## HARD CONSTRAINTS

#### You need to solve this problem (i.e. play this game) in 40 minutes total

- You need to create a log called "attempt_history.md"
- This log will have 2 column headers {attempt_index, attempt_time (mins:seconds), cumulative_attempt_time (mins:seconds)}
- You will initialize cumulative_attempt_time to 0 seconds before the GAME starts
- Then you will add attempt_time for a given attempt to cumulative_attempt_time
- You are allowed to play until the cumulative_attempt_time hits 40 minutes
- Cumulative_attempt_time = 40 minutes is the **HARD CONSTRAINT**

#### TPU MATMUL TEST

- You are REQUIRED TO WRITE A SMALL JAX MATRIX MULTIPLICATION SCRIPT THAT PRINTS DEVICE AND SPEED
- You are REQUIRED to RUN the script and CONFIRM THAT SPEEDS ARE "TPU-EXPECTED" NUMBERS
- ALWAYS CHECK THAT TPU IS CONNECTED AND RUNNING AT EXPECTED SPEEDS BEFORE PROCEEDING

#### You need to ALWAYS COMMIT AND PUSH EVERYTHING TO GIT

** You need to always commit and push the following to Git:
- attempt_history.md
- attempt_history_details.md
- attempt_history_details_n.md
- solve_segment_general_pinn_attempt_n.py
- dataset_name_plot_attempt_n.png (details for this file in subsequent portion of this md file)

#### WORK AUTONOMOUSLY

- The word SEMI-AUTONOMOUS has been clarified in this file
- You are REQUIRED TO WORK ALONE

## SOFT CONSTRAINTS

- Do your work as a background task
- Minimize permissions
- Try to work as autonomously as possible resolving small issues and deciding smaller issues that always come up in scientific computing and software project workflows

## TEST

- We are testing to see if CLAUDE CAN SOLVE THIS SEMI-AUTONOMOUSLY - VIVEK ALREADY HAS HIS VERSION and provided some HINTS that are JUST GENTLE GUIDANCE.
- Note that SEMI-AUTONOMOUS IS NOT A LICENSE FOR YOU TO PESTER THE HUMAN... SEMI-AUTONOMOUS MEANS SOMETHING VERY VERY SPECIFIC... YOU WORK AUTONOMOUSLY BUT VIVEK HAD PROVIDED HINTS THAT KINDA ARE PART OF THE SOLUTION... IT IS SEMI-AUTONOMOUS IN THAT SENSE - YOU WERE GIVEN SOME HINTS.. THAT'S ALL... BUT WORK AUTONOMOUSLY... HUMAN WILL PROVIDE NO HELP OR GUIDANCE FURTHER

## DELIVERABLE

### ONLY Plot: Angular Velocity Overlay (comprehensive)

- Three gyroscope components (ω_x, ω_y, ω_z) from the provided measurement data where each w_j is the "j"th row in a 3-row-1-col subplot-style plot (look for examples in @example_results subfolder)
- Note that the example plots in the @example_plots subfolder DO NOT HAVE THE DESIRED TITLE ENTRIES, THEY ONLY HAVE THE DESIRED OVERALL PLOT STYLE AND LAYOUT
- Three PINN-predicted angular velocities, reconstructed via ω = 2q*⊗q̇
- Translucent ±0.1 rad/s envelope around (ω_x, ω_y, ω_z)
- PINN-predicted bold dashed lines in a distinct color
- Overlaid on the same axes
- The plots NEED TO INCLUDE THE ERROR VALUE AND PASS/FAIL IN THE PLOT TITLE/HEADER
- THE plots for EVERY ATTEMPT NEED TO BE SAVED TO THE RESULTS FOLDER WITH the extension "datasetname_plot_attempt_n.png" for the "N"th attempt where datasetname = seg23 or trial_{idx} with idx = [000, 006, 009, 011, 016]

### Claude_explores folder

- If you want to explore things, create additional plots, etc... do so cleanly by writing to @claude_explores sub-subfolder that's inside the @claude_sandbox subfolder to start with (Unix Philosophy - clear separation of concerns)
- DO NOT POLLUTE RESULTS FOLDER with things THAT ARE NOT STRICT DELIVERABLES UNDER ANY CIRCUMSTANCES as this OVERWHELMS AND CONFUSES THE HUMAN WITHOUT REASON

- **Pass criteria**: the PINN dashed lines must stay inside the ±0.1 rad/s envelopes for ω_x, ω_y, ω_z) at every time point

## CLAUDE.md UPDATES (added again in case this doc undergoes a section-by-section deep discussion, this step naturally comes at the end in such a workflow)

- READ THIS FILE THOROUGHLY AND YOU ARE STRICTLY REQUIRED TO ENSURE THAT THE LOCAL PROJECT CLAUDE.MD IS UP TO DATE WITH THIS FILE

