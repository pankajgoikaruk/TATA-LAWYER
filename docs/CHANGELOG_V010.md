# v0.10 Documentation Update Changelog

Updated from supplied v0.9 documents to include:

1. v0.10 human-reviewed tri-state low-energy annotation.
2. Corrected final manual-review counts: 1,018 rows, 966 fully known, 52 partial, 277 silence positives, 741 silence negatives.
3. Masked manifest build output: 13,606 rows, 1,018 candidate-ID matches, 1,883 strict validation rows, 1,961 strict test rows.
4. New masked training modules and commands.
5. Latest masked fixed-threshold training results:
   - strict test Macro-F1 0.7950
   - Micro-F1 0.7952
   - Samples-F1 0.7655
   - Exact/Fully Known 0.5926
   - Hamming 0.0552
6. Comparison against original v0.9, full recovery, and positive-only ablation.
7. Updated interpretation: v0.10 improves annotation quality but not fixed-threshold strict-test performance.
8. Next step: per-label threshold tuning on strict validation.
