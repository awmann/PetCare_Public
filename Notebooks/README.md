# Notebooks

Worked examples, in order.

| notebook | needs | shows |
|---|---|---|
| [`01_fitting_a_transit_on_an_active_star.ipynb`](01_fitting_a_transit_on_an_active_star.ipynb) | nothing — synthetic, no network | why flattening first destroys the transit, the simultaneous variability + transit fit, the two variability models and how the fit arbitrates between them |
| [`02_running_the_tests.ipynb`](02_running_the_tests.ipynb) | the DS Tuc example's light curves | the command-line vocabulary, the result schema, chaining tests with `--use-fit`, reading the statuses |
| [`03_a_full_vetting_run.ipynb`](03_a_full_vetting_run.ipynb) | both examples' light curves + network | the driver over a candidate table, the summary, the HTML report, the cross-target tests |

Notebook 1 is committed with its outputs so it reads on GitHub without being
run. Notebooks 2 and 3 are committed clean. They run on the light curves in
[`../examples`](../examples), so they need no setup beyond the install; to use
your own data, change `LC_FILES` (notebook 2) or the candidate table and
`--lc-root` (notebook 3).

Each writes into `out_notebook/` or `out_run/` beside itself; both are
gitignored.
