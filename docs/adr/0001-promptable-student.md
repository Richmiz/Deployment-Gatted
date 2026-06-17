# ADR 0001: Promptable Student Scope

## Status

Implemented as an optional scaffold; not part of the default study path.

## Context

MedSAM is promptable: teacher predictions depend on box prompts. The current compact student is a plain image-to-mask U-Net trained from ground-truth masks and teacher soft masks. That makes the deployment story simple, but it also means the student is not promptable in the same way as the teacher.

The repository needs to be clear about this difference before describing student behavior under prompt changes.

## Decision

The default student remains the image-only `LightweightUNet`.

An optional `PromptableLightweightUNet` is available as a scaffold. It accepts the RGB image plus a fourth binary prompt-mask channel derived from a box prompt. This keeps the default training and comparison path stable while making prompt-conditioned student runs explicit when a user chooses that path.

## Consequences

- Default student results should be described as compact image-only segmentation, not promptable student inference.
- Prompt robustness outputs belong to MedSAM/LoRA teacher evaluations unless the promptable student path is explicitly trained and evaluated.
- Promptable-student outputs should be reported separately from the default image-only student outputs.

## Implementation Notes

- Default class: `LightweightUNet`
- Optional ablation class: `PromptableLightweightUNet`
- Prompt conversion helper: `box_prompt_to_mask`
