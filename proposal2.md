# Proposal 2: PB-DistillSAM

## Learning Prompt Response Basins for Prompt-Free Surgical Image Segmentation

### Working title

**PB-DistillSAM: Learning Prompt Response Basins for Prompt-Free Surgical Image Segmentation**

Alternative title:

**Beyond Prompt Embedding Mimicry: Counterfactual Prompt Basin Distillation for Surgical Image Segmentation**

---

## 1. Motivation

Distillation-SAM learns an automatic prompt representation by distilling prompt embeddings generated from ground-truth-derived prompts. This removes manual prompting at inference time, but the formulation implicitly treats the teacher prompt embedding as a point target.

This assumption is unnecessarily restrictive.

For a single surgical object, many different prompts can lead SAM to nearly identical and correct segmentations:

- a center positive point,
- a random interior positive point,
- multiple positive points,
- a positive-negative point combination,
- a tight bounding box,
- a slightly jittered bounding box,
- or a coarse mask prompt.

Therefore, a segmentation target does not correspond to one unique prompt embedding. Instead, it induces a **region of prompt space** whose members are functionally equivalent because they produce the same correct object mask.

At the same time, small prompt changes near difficult surgical structures can abruptly switch SAM to a wrong target, especially when instruments overlap, cross each other, are partially occluded, or touch visually similar tissue.

This proposal studies both sides of this behavior:

1. the region of prompts that remain valid for the target object, and
2. the closest counterfactual prompts that make SAM fail or switch to another structure.

We call this region the **Prompt Basin** and its boundary the **Prompt Failure Boundary**.

The central hypothesis is:

> A prompt-free student should not imitate one arbitrary teacher prompt embedding. It should learn to place its automatic prompt inside a robust region of SAM's prompt-response space and away from nearby failure boundaries.

---

## 2. Research question

Instead of asking:

> What is the correct prompt embedding for this image?

we ask:

> What region of SAM's prompt space consistently produces the correct surgical structure, where is the nearest failure boundary, and can that response geometry be distilled into a prompt-free model?

Formally, for image \(x\), target mask \(y\), frozen SAM image representation \(h(x)\), and prompt \(p\), define the decoder response

\[
F(x,p)=\mathrm{SAMDecoder}(h(x),p).
\]

A set of valid prompts is

\[
P^+(x,y)=\{p: \mathrm{Dice}(F(x,p),y)\ge \tau\}.
\]

These prompts form a task-conditioned prompt basin

\[
\mathcal{B}(x,y).
\]

Counterfactual prompts are nearby prompts that cause the decoder to leave this basin:

\[
P^-(x,y)=\{p: \mathrm{Dice}(F(x,p),y)<\tau\}.
\]

The student auto-prompt generator \(G(x)\) should learn a prompt representation \(z=G(x)\) that lies in a robust region of \(\mathcal{B}(x,y)\), not merely close to one sampled teacher embedding.

---

## 3. Why this is different from the current DistillSAM formulation

The current implementation already supports a strong auto-prompt distillation pipeline:

- frozen SAM image encoder,
- PVT-based cross adapter,
- automatically generated sparse and dense prompt embeddings,
- semantic or binary segmentation,
- revised semantic mask decoder,
- sparse prompt distillation,
- dense prompt distillation,
- attention distillation,
- segmentation supervision.

The current KD formulation mainly transfers prompt representations or decoder-side responses produced by selected teacher prompts.

Proposal 2 changes the learning target itself:

### Existing formulation

```text
GT mask
  -> sample teacher prompt
  -> SAM prompt encoder
  -> one teacher representation
  -> student embedding matching
  -> segmentation
```

### Proposal 2

```text
GT mask
  -> generate multiple valid prompts
  -> generate minimal counterfactual prompts
  -> evaluate all prompts with frozen SAM
  -> estimate prompt success basin and failure boundary
  -> distill basin response + boundary margin + local response geometry
  -> prompt-free student
```

The novelty is therefore not another backbone, adapter, or additional feature-matching loss. The method explicitly models the **geometry of SAM's prompt-conditioned behavior**.

---

## 4. Method overview

The method contains four main components:

1. **Valid Prompt Set Construction**
2. **Counterfactual Prompt Boundary Mining**
3. **Prompt Basin Distillation**
4. **Prompt-Response Geometry Distillation**

The current PVT cross-adapter student and frozen SAM backbone can remain largely unchanged.

```text
                        Ground-truth mask
                               |
              +----------------+----------------+
              |                                 |
      Valid Prompt Generator         Counterfactual Prompt Miner
              |                                 |
              +----------------+----------------+
                               |
                         Frozen SAM
                               |
                    Prompt-response samples
                               |
            +------------------+------------------+
            |                  |                  |
        Basin KD           Margin KD         Response KD
            |                  |                  |
            +------------------+------------------+
                               |
                     PVT Auto-Prompt Student
                               |
                         Frozen SAM Decoder
                               |
                           Segmentation
```

---

## 5. Valid Prompt Set Construction

For each target mask \(y\), generate a diverse prompt candidate set rather than a single prompt.

Candidate prompt families can include:

- center positive point,
- random interior positive point,
- boundary-near interior point,
- multiple positive points,
- positive + negative points,
- tight bounding box,
- translated or scaled bounding boxes,
- coarse foreground mask.

Let the candidate set be

\[
\tilde P=\{p_1,p_2,\ldots,p_K\}.
\]

Each prompt is passed through frozen SAM and validated using the ground-truth segmentation:

\[
P^+=\{p_i\in\tilde P:\mathrm{Dice}(F(x,p_i),y)\ge\tau\}.
\]

A practical initial threshold is

\[
\tau=0.90,
\]

with dataset-specific tuning if necessary.

Importantly, valid prompts are **not forced to have identical embeddings**. They only need to be functionally equivalent with respect to the segmentation target.

This prevents the model from treating an arbitrary prompt sample as the unique teacher solution.

---

## 6. Counterfactual Prompt Boundary Mining

Random negative prompts are easy and often uninformative. Surgical segmentation failures usually occur near semantically or spatially competing structures.

We therefore construct **minimal counterfactual prompts**.

Given a successful prompt \(p^+\), find a nearby prompt \(p^-\) such that SAM no longer segments the intended object:

\[
p^- = \arg\min_p d(p,p^+)
\]

subject to

\[
\mathrm{Dice}(F(x,p),y)<\tau.
\]

This estimates a local failure boundary in prompt space.

### 6.1 Point-prompt counterfactuals

For a positive point, gradually move the point along several directions until the response leaves the valid basin.

Examples:

- center -> object boundary -> surrounding tissue,
- instrument A -> contact region -> instrument B,
- visible instrument segment -> occluded region,
- foreground -> specular highlight,
- foreground -> blood/smoke region.

The closest failing point is retained as a hard counterfactual.

### 6.2 Bounding-box counterfactuals

Apply controlled transformations:

- translation,
- expansion,
- contraction,
- aspect-ratio distortion,
- one-side boundary shift.

For each transformation direction, search for the smallest perturbation that causes a large segmentation degradation or target switch.

### 6.3 Mask-prompt counterfactuals

Perturb coarse masks through:

- erosion,
- dilation,
- object-part removal,
- adjacent-object contamination,
- occlusion simulation.

Again, retain near-boundary failures rather than arbitrary corrupted masks.

---

## 7. Surgical Adjacent-Structure Counterfactuals

A surgical-specific contribution is to mine hard counterfactuals using object adjacency rather than generic background sampling.

Construct an image-level adjacency graph

\[
G=(V,E),
\]

where each node represents a semantic object or instrument instance and an edge connects spatially adjacent or overlapping structures.

Possible hard relationships include:

- instrument A <-> instrument B,
- instrument <-> tissue,
- instrument <-> vessel,
- instrument <-> iris,
- instrument <-> wound boundary,
- overlapping instruments,
- partially occluded instruments.

For target object \(i\), adjacent object \(j\) becomes a source of hard counterfactual prompts whenever

\[
\mathrm{dist}(M_i,M_j)<d
\]

or their boundaries overlap after a small dilation.

This makes the learned failure boundary clinically and visually relevant rather than dominated by easy background regions.

---

## 8. Prompt Basin Distillation

Let successful prompts produce decoder logits

\[
M_k^+=F(x,p_k^+).
\]

Instead of selecting one teacher response, compute a robust consensus representation of the successful basin.

A simple first version is

\[
\bar M^+=\frac{1}{|P^+|}\sum_{p\in P^+}F(x,p).
\]

The student auto-prompt is

\[
z=G(x).
\]

The basin loss encourages the student response to agree with the valid prompt response distribution:

\[
L_{basin}=D(F(x,z),\bar M^+),
\]

where \(D\) can initially be KL divergence, Dice loss, BCE, or a combination on logits/probabilities.

A stronger version can model uncertainty across successful prompts:

\[
\mu(x)=\mathbb{E}_{p\in P^+}[F(x,p)]
\]

and

\[
\sigma^2(x)=\mathrm{Var}_{p\in P^+}[F(x,p)].
\]

Pixels with high teacher agreement receive stronger supervision, while highly prompt-sensitive pixels receive weaker or uncertainty-aware weighting.

---

## 9. Counterfactual Margin Distillation

The student should not only reproduce successful responses; it should also remain separated from nearby failure responses.

Let

\[
M^+=F(x,p^+), \qquad M^-=F(x,p^-).
\]

Define response-space distance

\[
d_F(a,b)=D(F(x,a),F(x,b)).
\]

The student is encouraged to remain closer to valid responses than to failure responses:

\[
L_{margin}=\max\left(0,
 m+d_F(z,P^+)-d_F(z,P^-)
\right).
\]

This avoids relying purely on Euclidean distance between prompt embeddings, whose geometry is not guaranteed to correspond directly to segmentation behavior.

An embedding-space margin can still be evaluated as an ablation:

\[
L^{emb}_{margin}=\max\left(0,
 m+d(z,E^+)-d(z,E^-)
\right).
\]

The comparison between embedding-space and response-space margins is itself a useful experiment.

---

## 10. Prompt-Response Geometry Distillation

Successful and failing prompt samples provide more information than isolated outputs. They describe how SAM changes when the prompt moves.

For prompt \(p\) and a small perturbation \(\delta\), define the finite-difference response

\[
R_T(x,p,\delta)=
\frac{F(x,p+\delta)-F(x,p)}{\|\delta\|}.
\]

Instead of requiring a full Jacobian, sample a small set of meaningful perturbation directions:

- toward the closest target boundary,
- away from the target center,
- toward the closest competing object,
- box translation,
- box scale change,
- mask erosion/dilation.

The teacher therefore supplies a local prompt-response profile.

The student should reproduce the relevant response geometry around its predicted auto-prompt:

\[
L_{response}=D(R_S,R_T).
\]

This teaches the student not only where a good response exists, but also which directions are stable and which directions lead toward failure.

The contribution should be framed specifically as **prompt-response geometry distillation for SAM**, not generic Jacobian matching, because Jacobian-based knowledge transfer exists in earlier knowledge-distillation literature.

---

## 11. Full training objective

A minimal first implementation is

\[
L=
L_{seg}
+\lambda_bL_{basin}
+\lambda_mL_{margin}
+\lambda_rL_{response}.
\]

The original DistillSAM losses can optionally be retained:

\[
L=
L_{seg}
+\lambda_sL_{sparseKD}
+\lambda_dL_{denseKD}
+\lambda_aL_{attentionKD}
+\lambda_bL_{basin}
+\lambda_mL_{margin}
+\lambda_rL_{response}.
\]

However, the preferred experimental design is to add the new losses incrementally so that the contribution of prompt-basin learning is clearly measurable.

---

## 12. Recommended implementation stages

### Stage 1: Offline prompt-basin mining

Avoid making the first implementation unnecessarily expensive.

For each training image:

1. sample \(K\) prompt candidates,
2. run frozen SAM,
3. calculate Dice against ground truth,
4. store successful prompts,
5. search for a small number of nearby failures,
6. cache prompt metadata and teacher outputs.

Suggested first setting:

- \(K=8\) to \(16\) positive candidates,
- 2 to 4 counterfactual directions per successful prompt,
- retain the top 2 nearest failures,
- cache low-resolution SAM logits rather than full-resolution outputs.

This allows the new training objective to be tested without repeatedly querying many teacher prompts online.

### Stage 2: Basin KD

Implement only

\[
L_{seg}+L_{basin}.
\]

This is the cleanest test of the central hypothesis: multi-prompt functional supervision should outperform single prompt embedding imitation.

### Stage 3: Counterfactual margin

Add

\[
L_{margin}.
\]

Prioritize adjacent-object and boundary counterfactuals.

### Stage 4: Prompt-response geometry

Add finite-difference response supervision only after Stage 2 and Stage 3 show positive results.

This prevents an expensive component from obscuring the core contribution.

---

## 13. Integration with the current repository

The existing architecture can be reused:

```text
Frozen SAM Image Encoder
          |
          | intermediate features
          v
   PVT Cross Adapter
          |
    +-----+------+
    |            |
 sparse        dense
 prompt        prompt
    |            |
    +-----+------+
          |
   SAM / Revised Decoder
          |
     segmentation
```

The main additions should live on the teacher/training side.

Suggested modules:

```text
distsam/
  basin/
    __init__.py
    prompt_candidate_generator.py
    counterfactual_miner.py
    basin_builder.py
    response_geometry.py

  losses/
    basin_loss.py
    counterfactual_margin_loss.py
    response_geometry_loss.py

  tools/
    mine_prompt_basins.py
```

Suggested cached record format:

```python
{
    "image_id": "...",
    "target_class": 3,
    "positive_prompts": [...],
    "negative_prompts": [...],
    "positive_scores": [...],
    "negative_scores": [...],
    "teacher_logits": [...],
    "boundary_distances": [...],
}
```

The existing `teacher_prompt_builder.py` can be reused for initial prompt construction, but Proposal 2 should introduce a separate basin-mining abstraction rather than overloading the current single-teacher-prompt path.

---

## 14. Ablation study

A clean ablation sequence is essential.

| Variant | Seg | Original KD | Basin KD | Counterfactual Margin | Response Geometry |
|---|---:|---:|---:|---:|---:|
| Baseline | yes | no | no | no | no |
| DistillSAM | yes | yes | no | no | no |
| + Basin | yes | yes/no | yes | no | no |
| + Basin + CF | yes | yes/no | yes | yes | no |
| Full PB-DistillSAM | yes | yes/no | yes | yes | yes |

Report at least:

- Dice,
- IoU,
- foreground Dice,
- per-class Dice,
- boundary Dice,
- Hausdorff distance if applicable,
- performance on overlapping/occluded structures,
- performance under image-domain shift.

The key comparison is not only whether overall Dice improves, but whether the model becomes more robust on images where multiple nearby structures create ambiguous prompt responses.

---

## 15. New evaluation metric: Prompt Basin Radius

Standard Dice does not directly test whether the student has learned a robust location in prompt-response space.

Define **Prompt Basin Radius (PBR)** for a valid prompt \(p\) as the smallest perturbation required to cause segmentation failure:

\[
\mathrm{PBR}(p)=
\min_{\delta}\|\delta\|
\]

subject to

\[
\mathrm{Dice}(F(x,p+\delta),y)<\tau.
\]

For point prompts, distance can be normalized by image diagonal or target bounding-box size.

For boxes, perturbation can be defined over normalized center/width/height parameters.

For a student's auto-prompt \(z\), we expect

\[
\mathrm{PBR}(z_{PB-DistillSAM})
>
\mathrm{PBR}(z_{DistillSAM}).
\]

Interpretation:

> The proposed method should place the automatic prompt deeper inside a successful prompt basin rather than near a decision boundary.

This metric directly supports the paper's central claim.

---

## 16. Additional robustness evaluation

### 16.1 Synthetic occlusion

Artificially occlude parts of target instruments and test whether the auto-prompt remains associated with the same target.

### 16.2 Adjacent-object challenge subset

Create an evaluation subset containing cases with:

- touching instruments,
- crossing instruments,
- severe overlap,
- instrument-tissue contact,
- strong specular highlights,
- partial visibility.

### 16.3 Counterfactual switching rate

Measure how frequently a small perturbation causes SAM/student to switch to an adjacent object.

Define

\[
CSR=\frac{\#\text{target-switch failures}}{\#\text{tested perturbations}}.
\]

A robust auto-prompt should reduce CSR.

---

## 17. Novelty positioning

The proposal should avoid claiming novelty for concepts that already exist individually.

Do **not** position the contribution as simply:

- automatic prompt generation,
- multi-prompt consistency,
- feature distillation,
- attention distillation,
- Jacobian distillation,
- prompt perturbation analysis,
- or another SAM adapter.

Those directions already have substantial prior work.

The novelty claim should instead be the combination and problem formulation:

> We formulate prompt-free SAM adaptation as learning the geometry of a target-conditioned prompt-response basin. Rather than matching a single teacher prompt, the student is trained using the set of functionally valid prompts, minimally perturbed counterfactual prompts that identify the nearest failure boundary, and the decoder responses induced along these directions.

A second, surgical-specific contribution is:

> Hard counterfactuals are mined from spatially adjacent surgical structures, explicitly targeting instrument-instrument and instrument-tissue confusion rather than generic background negatives.

---

## 18. Distinction from closely related directions

### Distillation-SAM

Distills ground-truth-derived prompt information into an automatic prompting mechanism.

**Difference:** PB-DistillSAM treats a correct target as a region/set of functionally equivalent prompts and explicitly models the nearest failing prompts.

### Automatic prompt-generation methods

Generate points, boxes, prompt embeddings, or other prompt representations directly from image features.

**Difference:** the proposed contribution is not the prompt generator architecture. It is the supervision derived from the topology of SAM's prompt-response space.

### Prompt-consistency methods

Encourage similar predictions from different prompts.

**Difference:** PB-DistillSAM does not assume every prompt perturbation should be invariant. It explicitly learns both sides:

1. perturbations that should remain within the same target basin, and
2. perturbations that cross a semantic failure boundary.

### Prompt-sensitivity analysis

Studies how SAM performance changes under point or box perturbations.

**Difference:** PB-DistillSAM converts that sensitivity into a trainable teacher signal for a prompt-free student.

### Generic Jacobian matching

Transfers input-output derivatives between teacher and student networks.

**Difference:** response geometry is only one component. The main contribution is the SAM-specific construction of successful prompt basins and semantic counterfactual failure boundaries.

---

## 19. Main contributions

The paper should be presented with three primary contributions.

### Contribution 1: Prompt Basin Distillation

We reformulate auto-prompt distillation from single-embedding imitation into learning a set of functionally equivalent prompts that induce the same correct surgical segmentation.

### Contribution 2: Counterfactual Failure-Boundary Distillation

We construct minimally perturbed counterfactual prompts, with special emphasis on adjacent surgical structures, to explicitly identify and transfer SAM's local prompt failure boundary.

### Contribution 3: Prompt-Response Geometry Learning

We transfer how SAM's segmentation response changes along meaningful prompt perturbation directions, encouraging the prompt-free student to occupy robust regions of prompt space rather than merely approximate an arbitrary teacher embedding.

---

## 20. Core paper narrative

A concise framing for the introduction is:

> Existing auto-prompt distillation methods implicitly treat the teacher prompt embedding as a unique supervision target. However, a single object can be correctly segmented by many substantially different prompts, while a small perturbation near another surgical structure can abruptly redirect SAM to the wrong target. We therefore argue that the relevant object of distillation is not an individual prompt embedding but the geometry of SAM's prompt-response space. PB-DistillSAM learns the basin of prompts that remain functionally valid for the target and the nearest counterfactual boundary at which the decoder response fails or switches to an adjacent structure. This geometry is distilled into a prompt-free student, encouraging its learned prompt to reside in a robust region of the foundation model's prompt space.

---

## 21. Minimum viable experiment

The first experiment should test the hypothesis with the least additional complexity.

### MVP

1. Keep the current DistillSAM student architecture unchanged.
2. For each training mask, sample 8 valid prompt candidates.
3. Run frozen SAM and keep prompts with Dice >= 0.90.
4. Average successful teacher low-resolution logits.
5. Train with segmentation loss + basin consensus loss.
6. Compare against the current single-prompt KD baseline.

If this already improves generalization or difficult-class Dice, continue with counterfactual mining.

### MVP success criterion

The idea is supported if Basin KD produces at least one of the following without architecture changes:

- higher mean Dice/IoU,
- higher low-frequency class Dice,
- better boundary Dice,
- lower variance across random teacher prompts,
- better performance on overlapping/adjacent-object cases,
- larger Prompt Basin Radius.

If Basin KD does not outperform single-prompt KD, the counterfactual and response-geometry components should not be added until the cause is understood.

---

## 22. Main hypothesis

The central hypothesis to test is:

\[
\boxed{
\text{Learning the functional prompt basin is more informative than imitating a single prompt embedding.}
}
\]

The stronger hypothesis is:

\[
\boxed{
\text{Explicitly learning the nearest semantic failure boundary yields more robust prompt-free surgical segmentation.}
}
\]

This keeps Proposal 2 conceptually distinct from architecture-centric SAM adaptation and conventional knowledge distillation.