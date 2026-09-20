# OSAF-PWP

Canonical reusable software infrastructure for Pacific Warm Pool analysis.

This repository preserves the audited core methodology used for spherical, physical-area-weighted Pacific Warm Pool calculations, including Pacific-mask handling, spherical area weighting, centroid calculation, and quality-control infrastructure.

## Scientific Context

OSAF-PWP supports the computational workflow associated with:

**Machado, F. V.**
*Defining the Pacific Warm Pool: Threshold Dependence of Centroid Geometry and Robustness of Interannual Variance Modulation.*

Journal of Atmospheric and Oceanic Technology manuscript.

## Release

**Current archival software release:** v1.0.0

The v1.0.0 release contains the audited core software used for spherical, physical-area-weighted Pacific Warm Pool centroid calculation. Byte-integrity checks, clean/fresh-clone validation, automated tests, and GitHub Actions CI passed before release.

[View v1.0.0 release](https://github.com/blackbeltbjj/OSAF-PWP/releases/tag/v1.0.0)

## Canonical Program 05

`src/05_calculate_pwp_centroid.py`

SHA-256:

`729B80855247DE4F690C790DB3129262BB2E808FC43B9088D9E83B00E3DA3150`

## Reproducibility Role

OSAF-PWP provides the reusable computational core on which the threshold-sensitivity analysis is built.

**Scientific workflow:**

`NOAA OISST v2.1 -> Pacific mask -> spherical area weighting -> warm-pool geometry -> centroid calculation -> quality control -> derived scientific products`

## Related Repository

[pwp-threshold-centroid-sensitivity](https://github.com/blackbeltbjj/pwp-threshold-centroid-sensitivity) contains the threshold-sensitivity and robustness analyses supporting the manuscript.

## Archival DOI

**OSAF-PWP v1.0.0 - Zenodo Software Archive**

- Version DOI: https://doi.org/10.5281/zenodo.21964951
- GitHub release: https://github.com/blackbeltbjj/OSAF-PWP/releases/tag/v1.0.0

### Associated Paper 1 archives

The threshold-sensitivity software and frozen derived scientific products supporting the same study are archived separately:

- PWP Threshold-Centroid Sensitivity Reproducibility Package v1.0.0: https://doi.org/10.5281/zenodo.21964955
- Pacific Warm Pool Threshold-Centroid Sensitivity: Derived Data v1.0.0: https://doi.org/10.5281/zenodo.21976955

These persistent records preserve the released software and derived-data products associated with the Paper 1 reproducibility chain.

## Author

**Fabio Vieira Machado**
ORCID: [0000-0003-0723-075X](https://orcid.org/0000-0003-0723-075X)

## License

MIT License.
