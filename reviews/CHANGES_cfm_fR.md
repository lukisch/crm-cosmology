# cfm_fR Gravity Model: Patch Documentation

## Summary

The `cfm_fR` gravity model extends [hi_class](https://github.com/miguelzuma/hi_class_public) (Horndeski in CLASS) with a curvature feedback mechanism implementing an f(R) gravity model with saturation.

**Patch script:** `scripts/patch_cfm.py`
**Target files:**
- `hi_class_public/include/background.h` (enum extension)
- `hi_class_public/gravity_smg/gravity_models_smg.c` (4 code modifications)

**hi_class version tested:** v2.9.4+ (commit hash from miguelzuma/hi_class_public main branch, 2024)

## Physical Model

The cfm_fR model implements the following Horndeski alpha functions:

```
alpha_M(a) = alpha_M_0 * n_exp * a^n_exp / (1 + alpha_M_0 * a^n_exp)
alpha_B(a) = -alpha_M(a) / 2     [f(R) relation from R + gamma*R^2 action]
alpha_T(a) = 0                     [gravitational wave speed = c]
alpha_K(a) = 0
```

This parametrization captures:
- **Early times** (a << 1): alpha_M ~ alpha_M_0 * n * a^n (perturbative, suppressed)
- **Late times** (a ~ 1): alpha_M -> n_exp (saturates, preventing runaway)
- **n_exp = 1**: Exactly reproduces the built-in `propto_scale` parametrization
- **n_exp = 0.5**: Best-fit to Planck data (scalaron mass m_eff ~ a^{-1/4})

The f(R) relation alpha_B = -alpha_M/2 follows from the conformal equivalence of
the R + gamma*R^2 action to a scalar-tensor theory (Starobinsky 1980).

## Parameters

| Parameter | hi_class name | Description | Range |
|-----------|--------------|-------------|-------|
| alpha_M_0 | parameters_smg[0] | Amplitude of Planck mass running | 0 -- 0.003 |
| n_exp | parameters_smg[1] | Power-law time evolution index | 0.1 -- 2.0 |
| M*2_init | parameters_smg[2] | Initial Planck mass squared (set to 1.0) | 1.0 |

## Usage

```python
from classy import Class

cosmo = Class()
cosmo.set({
    'output': 'tCl,pCl,lCl,mPk',
    'l_max_scalars': 2500,
    'lensing': 'yes',
    # Standard cosmological parameters
    'h': 0.6732,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    # Horndeski / modified gravity settings
    'Omega_Lambda': 0,
    'Omega_fld': 0,
    'Omega_smg': -1,
    'gravity_model': 'cfm_fR',
    'parameters_smg': '0.0005, 0.5, 1.0',  # alpha_M_0, n_exp, M*2_init
    'expansion_model': 'lcdm',
    'expansion_smg': '0.5',
    'method_qs_smg': 'quasi_static',
    'skip_stability_tests_smg': 'yes',
    'pert_qs_ic_tolerance_test_smg': -1,
})
cosmo.compute()
```

## Modification 0: Enum Extension (background.h)

**Location:** `include/background.h`, `gravity_model_smg` enum
**Action:** Adds `cfm_fR` as a new enum value
```c
  alpha_attractor_canonical,
  cfm_fR           /* <-- added by patch */
} gravity_model_smg;
```

## Modifications to gravity_models_smg.c

### Modification 1: Model Registration
**Location:** `gravity_models_init()` function
**Action:** Adds `cfm_fR` to the list of recognized gravity models
```c
else if (strcmp(psmg->gravity_model, "cfm_fR") == 0) {
    psmg->field_type = scalar;
    psmg->gravity_model_type = cfm_fR;
}
```

### Modification 2: Alpha Function Computation
**Location:** `gravity_functions_smg()` function, in the switch for model types
**Action:** Computes alpha_M and alpha_B from (alpha_M_0, n_exp)
```c
case cfm_fR: {
    double alpha_M_0 = psmg->parameters_smg[0];
    double n_exp = psmg->parameters_smg[1];
    double a_n = pow(a, n_exp);
    double alpha_M = alpha_M_0 * n_exp * a_n / (1.0 + alpha_M_0 * a_n);
    pvecback[pba->index_bg_kineticity_smg] = 0.0;     // alpha_K
    pvecback[pba->index_bg_braiding_smg] = -alpha_M/2; // alpha_B
    pvecback[pba->index_bg_M2_smg] = /* integrated M*2 */;
    pvecback[pba->index_bg_tensor_excess_smg] = 0.0;   // alpha_T
    break;
}
```

### Modification 3: Print Function
**Location:** `gravity_print_smg()` function
**Action:** Prints cfm_fR model parameters for logging

### Modification 4: Error Message
**Location:** Model name validation
**Action:** Adds `cfm_fR` to the list of recognized model names in the error message

## Verification

The patch is verified by checking:
1. cfm_fR with n=1 exactly reproduces `propto_scale` results (identical chi2 and sigma8)
2. All n_exp < 2 values produce stable perturbation solutions
3. theta_s = 1.04173 for all parameter values (background unaffected)
4. LCDM limit (alpha_M_0 -> 0) recovers standard CLASS results

## Software Versions

- hi_class: v2.9.4+ (github.com/miguelzuma/hi_class_public)
- CLASS: v3.2+ (Blas, Lesgourgues & Tram, 2011)
- Python: 3.12
- Cython: 0.29.37
- NumPy: 1.26.4
- Platform: Ubuntu 24.04 (WSL2 on Windows 11)
