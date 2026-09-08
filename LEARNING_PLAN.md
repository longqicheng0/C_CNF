# Continuous normalizing flows from scratch in C

Build a small numerical learning system in C, then use it to train a continuous normalizing flow (CNF). The final program should generate samples from a learned two-dimensional distribution, evaluate log densities, and save enough information to reproduce its results.

This is a proposed implementation and learning plan. The milestones below are not implemented yet. An MVP is complete when its executable demonstrates the intended behavior and its correctness checks pass.

Assumptions: C17, CPU execution, double precision, and the C standard library plus the platform math library. You implement the numerical operations, differentiation, network, optimizer, ODE solver, probability calculations, and output generation. Compiler/debugger/build tools remain normal development tools. Functions such as exp, log, sqrt, and tanh may come from the math library.

Keep the first model two-dimensional, with small dense layers and fixed-step integration. Use ordinary arrays and loops. Add infrastructure when a milestone needs it; a general tensor framework, GPU backend, and custom transcendental-function library are separate future projects.

1. **MVP 1: A reliable C executable and memory model**

   Learn compilation/linking, header files, structs, pointers, array indexing, stack versus heap storage, allocation lifetime, and basic debugging.

   Build a small executable, a Makefile, and an assertion-based test executable. Support debug and release builds. Use explicit array dimensions, size_t for sizes, and clear ownership: the component allocating a buffer owns its release unless documented otherwise. Introduce only the small dynamic buffers needed now.

   Start debug builds with C17, warnings, debug symbols, and AddressSanitizer/UndefinedBehaviorSanitizer. Clang documents the relevant flags and checks in its [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) and [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html) guides.

   Done when: allocation, initialization, use, and cleanup work without sanitizer reports; malformed dimensions are rejected; debug and release builds both run. The executable prints the result of a small, hand-checkable array calculation.

2. **MVP 2: Numerical primitives and a model that learns**

   Learn dot products, matrix-vector multiplication, mean squared error, derivatives, and gradient descent.

   Implement contiguous row-major storage, dot products, matrix-vector multiplication, a deterministic seeded random generator, and a small CSV writer. Fit y = 2x + 1 using manually derived gradients for w and b. Add a central finite-difference helper for checking derivatives, with several perturbation sizes.

   Done when: arithmetic matches hand-worked examples, w and b converge near 2 and 1 on a well-scaled dataset, analytic gradients agree with finite differences, and a loss CSV shows learning. A fixed seed reproduces the random sequence in the same build/environment.

   Scope: matrix-vector operations are enough for the first network; batching can initially be a loop over samples.

3. **MVP 3: A scalar reverse-mode differentiation engine**

   Learn computational graphs, the chain rule, gradient accumulation, and reverse traversal. Use [Karpathy's micrograd](https://github.com/karpathy/micrograd) as a conceptual reference and implement the behavior in C.

   Store graph nodes in an append-only tape. Each node records its value, gradient, operation, and parent node indices. Use indices instead of pointers into a growable array so reallocating the tape does not invalidate graph links. Append each operation after its parents, then traverse backward in reverse insertion order. Start with constants/parameters, addition, subtraction, multiplication, tanh, exp, and log as needed by examples.

   Keep persistent parameter values outside the temporary tape. Create parameter leaves for each evaluation, copy their gradients back after backward, then reset/reuse tape storage. Node indices are valid only for their current tape lifetime.

   Done when: gradients match finite differences, including shared operands such as x*x + x; all parent contributions accumulate; repeated training steps do not retain old graphs or old gradients; the linear learner now works through automatic differentiation.

4. **MVP 4: A neural network built on your tape**

   Learn affine layers, nonlinear activations, initialization, forward evaluation, and parameter updates.

   Implement a DenseLayer and an MLP with contiguous parameter storage. Express dense operations as loops over tape operations. Begin with a 2 -> 4 -> 1 network, tanh activations, and XOR targets -1/+1. Use SGD and mean squared error.

   Done when: the four XOR cases are classified correctly by output sign; a tiny network's parameter gradients pass finite-difference checks; parameter counts and dimensions are explainable; the same example can be run repeatedly without memory growth from retained tapes.

   XOR proves the network can fit a nonlinear relationship. It does not measure generalization.

5. **MVP 5: A reusable training program**

   Learn minibatches, training/validation splits, learning rates, Adam's moment estimates and bias correction, and reproducibility.

   Implement a noisy two-moons generator, shuffling, SGD and then Adam, training/validation metrics, and model save/load. Generate Gaussian noise from your seeded uniform generator, for example with Box-Muller; make its log input strictly positive. Keep optimizer buffers separate from graph storage. Start with one sample's tape at a time, accumulate parameter gradients into a separate batch buffer, divide by batch size, and update once per batch.

   Emit CSV and a simple SVG scatter/decision plot directly from C. Save a versioned model format with dimensions and parameter values; avoid serializing raw structs with pointers or compiler-dependent padding. Save optimizer state too if promising exact training resumption.

   Done when: held-out classification beats a majority-class baseline; save/load reproduces predictions; batch gradients match the sum/mean of individual sample gradients; repeated minibatches reuse bounded tape storage.

6. **MVP 6: Probability and an invertible affine flow**

   Learn probability density, Gaussian log density, maximum likelihood, and change of variables. Learn the Jacobian as local sensitivity and its determinant as local volume scaling.

   Implement standard Gaussian sampling/log density and Gaussian parameter fitting. Build the one-dimensional transform x = exp(s)*z + b, its inverse, and its density correction:

   log p_x(x) = log p_z((x-b)*exp(-s)) - s.

   Fit s and b from generated observations through your tape. Add a two-dimensional Gaussian-mixture generator with known component parameters; this will supply the CNF capstone data. Use log-sum-exp when evaluating mixture log density.

   Done when: affine round trips recover inputs, fitted mean and scale agree with sample statistics, transformed samples agree with the predicted distribution, and the density calculation matches the analytic Gaussian answer.

   An expressive discrete coupling flow is an optional extra here. The affine example establishes the prerequisite without adding another large model.

7. **MVP 7: A numerical ODE solver**

   Learn initial-value problems, velocity versus position, step size, accumulated numerical error, and the Euler/RK4 formulas.

   Implement Euler and fixed-step RK4 first using ordinary doubles and explicit work buffers. Solve z' = a*z, then a two-dimensional rotation. Support both integration directions with a signed step. Save trajectories as CSV and SVG.

   Done when: results converge toward z(t) = exp(a*t)*z(0) as the step is refined in a sensible numerical range; rotation preserves radius increasingly well with refinement; forward-then-backward integration approximately recovers the input.

   ODE inversion is numerical. A forward RK4 step followed by a backward RK4 step is not an exact algebraic inverse.

8. **MVP 8: Differentiation through an ODE and a learned velocity field**

   Learn how repeated integration steps become one larger computational graph. Network parameters are shared across all times and all RK4 stages.

   Implement the same RK4 arithmetic using tape node indices for the evolving state. Keep the complete sample trajectory's computation on the tape until backward finishes. Treat solver time and fixed step size as constants. First learn the scalar a in z' = a*z from endpoint observations, then use a small MLP to learn damped-rotation trajectories.

   Done when: the loss gradient with respect to a agrees with finite differences of the same discrete solver; the learned parameter approaches the expected value as solver error is reduced; the MLP predicts trajectories from held-out starting points within the training region.

   Do not convert graph-valued intermediate states into new constant leaves between steps. That would discard parameter dependencies. Record one shared parameter leaf per parameter for a sample evaluation and reuse it throughout the solve.

9. **MVP 9: Spatial derivatives that remain differentiable**

   Learn the spatial Jacobian, its trace/divergence, and why likelihood training needs derivatives of derivatives. A standard micrograd backward pass writes numerical gradients; that alone does not keep the divergence computation differentiable with respect to parameters and evolving positions.

   Build a small Dual2 abstraction with three tape node indices: value, derivative with respect to local x, and derivative with respect to local y. Propagate the two spatial derivatives forward through the network, expressing every derivative formula as ordinary tape operations. For h = tanh(u), use d_i h = (1-h*h)*d_i u, entirely on the tape.

   At every velocity evaluation, initialize local position tangents to the identity basis. Time and parameter spatial tangents are zero. Their value components retain the existing tape dependencies. Form divergence as output_vx.dx + output_vy.dy. Reverse differentiation of this resulting graph produces the required parameter and state derivatives.

   This is a deliberately small composition of differentiation modes, rather than a promise of a general higher-order AD API. The [AD survey](https://www.jmlr.org/papers/v18/17-468.html) explains the underlying modes; [FFJORD's reference implementation](https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py) preserves a derivative graph when forming divergence for training.

   Done when: network Jacobians match central differences in x/y; gradients of divergence match finite differences in selected parameters AND state coordinates. Test a nonlinear example such as v_x = w*tanh(x), so a missing second-derivative term cannot hide behind constant divergence.

   The spatial derivatives are local derivatives at each field evaluation. Do not carry a single spatial tangent from the initial sample through the entire ODE and mistake the resulting trajectory sensitivity for local divergence.

10. **MVP 10: An analytically checked continuous density calculation**

    Learn the CNF density equation and adopt one fixed direction convention: Gaussian noise at t=0 and observed/generated data at t=1.

    Integrate z' = v(z,t) together with q' = -div(v). For likelihood evaluation, initialize z(1)=x and q(1)=0, integrate backward to 0, and calculate:

    log p_1(x) = log p_0(z(0)) - q(0).

    Here q is a density-change accumulator, not an absolute log density with an unknown initial value. For forward density evaluation, start from known Gaussian log density and integrate its derivative -div(v).

    Begin with v(z)=a*z in two dimensions. Its forward map is z(1)=exp(a)*z(0), and log density changes by -2*a along a trajectory. These follow the [instantaneous change-of-variables equation](https://arxiv.org/html/1806.07366v5#S4).

    Done when: density values and parameter gradients agree with analytic/finite-difference answers under solver refinement; forward/backward density calculations agree within numerical accuracy; expansion lowers density. Include a nonlinear velocity field to exercise the MVP 9 derivatives inside the complete solver.

    A concrete analytic check for the 2D linear field, at a fixed observed point x and integration interval [0,1], is:

    NLL(a; x) = log(2*pi) + 0.5*||x||^2*exp(-2*a) + 2*a.

    dNLL/da = 2 - ||x||^2*exp(-2*a).

    Finite differences of your discrete program should agree with its AD gradients; both should approach these continuous analytic answers as the step is refined.

11. **MVP 11: Train a complete 2D CNF**

    Use a standard 2D Gaussian base and a small tanh velocity network, initially 3 -> 8 -> 8 -> 2, with inputs x/y/t and two velocity outputs. The network predicts velocity; integration defines the transformation.

    Generate a fixed training and validation set from a small Gaussian mixture with nonzero component variances. Minimize mean negative log likelihood using the backward calculation above. Begin with small minibatches accumulated one sample at a time. Check gradients with a tiny network and a short solve before larger training runs. Increase hidden width or integration steps only as measurements justify it.

    Keep network weights fixed while collecting the batch's sample gradients. Update afterward. Reuse one sample's tape memory across the batch. Log tape node count/peak bytes as well as loss and runtime. Keep the state and log-density accumulator coupled through every RK4 stage.

    Done when: held-out NLL improves over a fitted Gaussian baseline; generated samples capture the target's modes; learned density contours and generated sample concentration agree; smaller solver steps give materially consistent results. Report sample quality and likelihood separately.

    The two-dimensional target has a proper continuous density. A regular invertible flow preserves dimension; use noisy moons if switching to a moons target rather than a noiseless curve.

12. **MVP 12: A reproducible C capstone**

    Build final train, sample, and evaluate entry points with saved configuration, model parameters, dataset/random seeds, and solver settings. Generate sample plots, density contours, trajectories, loss curves, and a short result summary directly from the C programs.

    Done when: a clean build can reproduce a documented run in the same environment; loading the model reproduces inference; a held-out report includes baseline/model NLL, round-trip error, solver refinement results, runtime, and peak tape memory; the numerical and memory checks pass.

    This completes the first implementation. The scalar tape is an educational reference whose performance you can measure before changing it.

The intended module ownership is:

| Module, added when needed | Responsibility |
| --- | --- |
| numeric.c / rng.c | Array operations, finite differences, seeded sampling |
| ad.c | Tape storage, scalar operations, reverse differentiation |
| nn.c | Dense layers, network parameter layout, evaluation |
| optim.c | SGD, Adam, persistent optimizer state |
| data.c / probability.c | Toy datasets, Gaussian and mixture log densities |
| ode.c | Plain-double and tape-based fixed-step integration |
| dual2.c | Local spatial derivatives represented on the tape |
| cnf.c | Coupled position/density ODE, likelihood, generation |
| io.c / plot.c | Versioned checkpoints, CSV, basic SVG output |

Put declarations under include/, implementations under src/, one executable example per milestone under examples/, numerical checks under tests/, and generated results under outputs/. These are proposed files, not existing implementation. Add each as its first caller appears.

The first working batch should be four small pieces: a compiling main.c, a Makefile, one assertion-based numerical check, and the manual linear-regression example. Work through that before writing the tape.

After the core capstone, choose one extension at a time:

| Extension | Bounded MVP |
| --- | --- |
| Performance | Profile tape allocation/traversal and matrix operations; improve the measured bottleneck while retaining the scalar reference checks. |
| Adaptive integration | Add a solver with an error estimate; compare accuracy and cost with fixed-step RK4. |
| Trace estimation | Add Hutchinson estimation as in [FFJORD](https://arxiv.org/abs/1810.01367); compare repeated estimates with the exact 2D trace before moving to larger dimensions. |
| Conditional CNF | Add a context input and verify generated distributions for each condition. |
| Flow matching | Train the velocity network on sampled interpolation targets, then sample with your ODE solver; compare against likelihood training on the same data. |
| Adjoint/checkpoint methods | Compare memory use and gradients against direct differentiation through the established solver. |

[Flow matching](https://arxiv.org/abs/2210.02747) is also an optional earlier generation milestone after MVP 8: it trains a CNF without ODE simulation in the training objective, and sampling still uses an ODE. It can provide a first visual generative result before implementing likelihood training. The core sequence above retains the density machinery because understanding and implementing CNF likelihood is part of this ground-up path.
