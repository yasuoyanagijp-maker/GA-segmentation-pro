"""CLAHE / foveal-correction ablation notes (GA_HRA_cases).

Measured on 3 Spectralis FAF images with scale≈10.46 um/px.

Findings
--------
1. CLAHE ON (clip=2) is essential.
   Turning CLAHE OFF enlarged masks by ~200–600% (false positives explode).
   Therefore use_clahe defaults to True; OFF is offered only for experiments.

2. Raising CLAHE clip from 2 → 4/6 often shrinks area on real HRA images
   (e.g. -23% to -67% vs clip=2), but synthetic soft-block images can grow.
   Treat clip as a user option, not a guaranteed shrinker.

3. Foveal correction at 50–100% may look intensity-ineffective when the
   fovea falls inside protected GA, yet still change Otsu thresholding /
   mask area. Evaluations must include mask-area delta (mask_effective),
   not only center brightness lift.

4. User-specified rectangular clear remains the reliable fail-safe when
   auto correction / CLAHE still leave over-large masks.
"""
