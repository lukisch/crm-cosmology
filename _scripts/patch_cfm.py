#!/usr/bin/env python3
"""Patch hi_class gravity_models_smg.c to add cfm_fR model"""

filepath = '/home/hi_class/gravity_smg/gravity_models_smg.c'

with open(filepath, 'r') as f:
    content = f.read()

changes = 0

# 1. Add model registration block (before the quintessence prefix check)
old_reg = '  if (strncmp("quintessence", string1, strlen("quintessence")) == 0) {'
new_reg = '''  if (strcmp(string1,"cfm_fR") == 0) {
     pba->gravity_model_smg = cfm_fR;
     pba->field_evolution_smg = _FALSE_;
     pba->M2_evolution_smg = _TRUE_;
     flag2=_TRUE_;
     pba->parameters_2_size_smg = 3;
     class_read_list_of_doubles("parameters_smg",pba->parameters_2_smg,pba->parameters_2_size_smg);
   }

  if (strncmp("quintessence", string1, strlen("quintessence")) == 0) {'''

if old_reg in content:
    content = content.replace(old_reg, new_reg, 1)
    changes += 1
    print("1. Model registration block added")
else:
    print("1. ERROR: Could not find registration insertion point")

# 2. Add alpha computation block (after eft_alphas_power_law, before eft_gammas)
old_bridge = '  else if ((pba->gravity_model_smg == eft_gammas_power_law) || (pba->gravity_model_smg == eft_gammas_exponential)) {'
new_bridge = '''  else if (pba->gravity_model_smg == cfm_fR) {
    /* CFM f(R)-type gravity model
     * Parameters: alpha_M_0 (amplitude), n_exp (power law exponent), M2_ini (initial Planck mass)
     * alpha_M(a) = alpha_M_0 * n_exp * a^n_exp / (1 + alpha_M_0 * a^n_exp)
     * f(R) relation: alpha_B = -alpha_M/2, alpha_T = 0, alpha_K = 0
     */
    double aM_0  = pba->parameters_2_smg[0];
    double n_exp = pba->parameters_2_smg[1];

    double a_pow_n = pow(a, n_exp);
    double alpha_M = aM_0*n_exp*a_pow_n/(1. + aM_0*a_pow_n);
    double alpha_B = -alpha_M/2.;

    pvecback[pba->index_bg_kineticity_smg] = 0.;
    pvecback[pba->index_bg_braiding_smg] = alpha_B;
    pvecback[pba->index_bg_tensor_excess_smg] = 0.;
    pvecback[pba->index_bg_M2_running_smg] = alpha_M;
    pvecback[pba->index_bg_delta_M2_smg] = delta_M2;
    pvecback[pba->index_bg_M2_smg] = 1.+delta_M2;
  }

  else if ((pba->gravity_model_smg == eft_gammas_power_law) || (pba->gravity_model_smg == eft_gammas_exponential)) {'''

if old_bridge in content:
    content = content.replace(old_bridge, new_bridge, 1)
    changes += 1
    print("2. Alpha computation block added")
else:
    print("2. ERROR: Could not find alpha insertion point")

# 3. Add print_stdout case (before default)
old_print = '    default:\n      printf("Modified gravity: output not implemented in gravity_models_print_stdout_smg() \\n");'
new_print = '''    case cfm_fR:
      printf("Modified gravity: cfm_fR (CFM f(R)-type) with parameters: \\n");
      printf(" -> alpha_M_0 = %g, n_exp = %g, M_*^2_init = %g \\n",
	      pba->parameters_2_smg[0],pba->parameters_2_smg[1],pba->parameters_2_smg[2]);
    break;

    default:
      printf("Modified gravity: output not implemented in gravity_models_print_stdout_smg() \\n");'''

if old_print in content:
    content = content.replace(old_print, new_print, 1)
    changes += 1
    print("3. Print stdout case added")
else:
    print("3. ERROR: Could not find print insertion point")

# 4. Update error message
old_err = "'alpha_attractor_canonical' ..."
new_err = "'alpha_attractor_canonical', 'cfm_fR' ..."

if old_err in content:
    content = content.replace(old_err, new_err, 1)
    changes += 1
    print("4. Error message updated")
else:
    print("4. ERROR: Could not find error message")

with open(filepath, 'w') as f:
    f.write(content)

print(f"\nDone: {changes}/4 modifications applied successfully!")
