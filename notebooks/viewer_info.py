"""
Plain-English text for the inversion explorer (export_viewer_inversion.py writes it into index.json).
Keys: tab.<name>, method.<family>, param.<name>, case.<id>, note.<topic>. Edit freely; re-run the export.
"""
INFO = {
    # ------------------------------------------------------------------ tabs
    'tab.explore': (
        'Change the model and watch the answer move. The VIS map, its uncertainty, the misfit and the '
        'evidence are recomputed in your browser for every setting. The model is: VIS = a mean level (or '
        'one level per province class, plus the covariates you tick) + a smooth random part. Three numbers '
        'control it: how noisy the data are taken to be (h), how much VIS varies (s), and over what '
        'distance it varies smoothly (length). "Best fit" finds the h and s the data prefer at this length. '
        'Positivity is not imposed here; the Compare tab has the positive versions.'),
    'tab.compare': (
        'Any maps side by side, sharing one view and one colour range. Pick a case, then a result for each '
        'panel. Drag to pan, scroll to zoom (all panels follow), hover a cell to read its value in every '
        'panel. "Difference" shows panel A minus panel B. Below each VIS map: the field it predicts at the '
        'data points, and what is left over.'),
    'tab.hypotheses': (
        'Which description of Australia does the satellite field prefer? Each bar is one hypothesis: '
        'constant VIS per province class, VIS following crustal thickness, the merging models, and so on. '
        'Longer bars are worse; the scale is the log evidence (a difference of 5 is strong, 2 is weak). '
        'Two versions: with a smooth random rest (fairer to simple hypotheses) and without it. Read with '
        'the caution in the null test: differences between covariate hypotheses are overstated.'),
    'tab.bounds': (
        'What do the data actually force? For each quantity (the mean VIS of a craton, or the craton-minus-rest '
        'contrast) the bar spans every value that some model fitting the data allows, with VIS between 0 '
        'and 4 SI.km. Darker bars add a smoothness limit. With positivity alone almost anything fits: the '
        'level of a large region is nearly invisible to a satellite (Runcorn\'s theorem). The intervals '
        'shrink only once models must be smooth, so the smoothness assumption, not the data, sets the '
        'levels. The dot is the smooth-model (GP) estimate with its own, much narrower, error bar.'),
    'tab.synthetic': (
        'Tests where the answer is known. Each synthetic case has a true VIS map; its field plus realistic '
        'noise was inverted like the real data. Compare every method with the truth: the error and the '
        'correlation are shown per panel.'),
    'tab.inputs': (
        'What goes in. The cells (about 100 km HEALPix pixels), the province classes and covariates used by '
        'the regression, the H&M VIS model (used only for its mean), and the satellite data: LCS-1 Br at '
        'the Earth\'s mean radius at the data points (about 55 km apart), degrees 16 to 133, with the field '
        'of everything outside the region removed. LCS-1 minus MF7 is the noise estimate.'),
    # ------------------------------------------------------------------ methods
    'tab.degrees': (
        'The same inversions repeated with other spherical-harmonic degree ranges and another field model. '
        'Each row is one run: LCS-1 (satellite only, to degree 185) or MLI 2E (satellite plus aeromagnetic '
        'and marine data, here cut at the same degrees), from degree 16 (below that the core field hides the '
        'crust) up to 100, 133 or 185, or with the lowest degrees removed (from 20 or 25). The chart shows, '
        'for one regional mean, the range every model that fits the data allows (bars) and the smooth GP '
        'estimate (dot, ±2 sd). For the synthetic cases the bar chart gives each run\'s map error against '
        'the truth. Pick two runs to map them side by side with their difference.'),
    'method.band': 'A map from one run of the degree-range comparison (Degrees tab).',
    'note.degrees': (
        'Degree l corresponds to a wavelength of about 40,000/l km: degree 16 is 2,500 km, 133 is 300 km, '
        '185 is 215 km. The two field models agree well over Australia to about degree 100 and less and less '
        'above it (correlation 0.81 for degrees 101-133, 0.65 for 134-185), so above 100 the choice of model '
        'matters as much as the number of degrees. The noise level for every run is the disagreement of the '
        'two models degree by degree (LCS-1\'s own damping of the highest degrees is not counted as error). '
        '"Synthetic only" runs test what degrees 1-15 would add if the core field did not hide them: '
        'they have no real data.'),
    'method.truth': 'The VIS map used to make the synthetic data.',
    'method.gp': (
        'Gaussian-process map: the most likely VIS when VIS is a mean level plus a smooth random part whose '
        'size and length are chosen by the data (maximum evidence). Can go slightly negative.'),
    'method.GP sd': 'One standard deviation of the GP map: where the data constrain VIS least, this is large.',
    'method.GP MAP': 'The GP map with VIS forced to be zero or more.',
    'method.l2': (
        'Smooth regularised map (Tikhonov / L2): penalises VIS gradients and departures from the H&M mean, '
        'with the penalty weight set so the misfit matches the noise. Larger beta = smoother.'),
    'method.tv': (
        'Blocky regularised map (total variation): penalises the size of VIS jumps, not their square, so it '
        'prefers flat patches with sharp edges. Same misfit rule as L2.'),
    'method.merging': (
        'Domains by merging: cells are merged into domains of constant VIS until merging more costs too much '
        'misfit. lambda: cost of each extra domain; gamma: cost of boundary length.'),
    'method.voronoi': 'Domains as Voronoi regions around seeds, sampled by MCMC; the map is the posterior mean.',
    'method.lvoronoi': 'Labelled Voronoi: a domain may consist of several Voronoi pieces; posterior mean.',
    'method.partition': 'Partition sampler: any grouping of cells into domains, sampled by MCMC; posterior mean.',
    'method.boundary probability': 'How often a domain boundary runs along each cell (0 never, 1 always).',
    'method.bounds': (
        'Extremal model: the most extreme VIS map (smallest or largest value of the quantity named) that '
        'still fits the data. It shows what the data cannot rule out, not a plausible Earth.'),
    'method.covariate': 'Input map, averaged over each cell.',
    # ------------------------------------------------------------------ parameters
    'param.h': (
        'Noise level h, in units of the LCS-1 minus MF7 difference (after weighting). Larger h = the data '
        'are trusted less, the map gets smoother and more uncertain.'),
    'param.s': 'Standard deviation s of the smooth random part of VIS (SI.km): how far VIS may stray from the mean.',
    'param.ell': (
        'Correlation length: over what distance the random part of VIS changes. Short = patchy maps, long = '
        'broad regional trends.'),
    'param.columns': (
        'Design columns: what the systematic part of VIS may follow. Class: one mean per province class '
        '(replaces the single mean). Covariates are standardised: the coefficient is the VIS change per '
        'standard deviation of the covariate.'),
    'param.beta': 'Penalty weight of the L2 / TV maps, as a multiple of the value at which the misfit equals the noise.',
    'param.evidence': (
        'Log evidence: how probable the data are under a model, after averaging over its unknown values. It '
        'rewards fit and penalises flexibility. Only differences matter.'),
    'param.chi2': 'Misfit: the sum of squared weighted residuals (845 independent data combinations).',
    # ------------------------------------------------------------------ cases
    'case.cov_truth': (
        'Synthetic: VIS = a mean per province class + 0.12 per sd of crustal thickness - 0.10 per sd of heat '
        'flow + a smooth random part. The regression should recover these numbers.'),
    'case.hast': (
        'Synthetic: one random VIS value per Hasterok province (53 provinces), no covariate effect. Values '
        'drawn from the spread of the H&M model, scaled so the field is as strong as the real one.'),
    'case.hast_var': 'Synthetic: as the province case, plus smooth variation within provinces (25%).',
    'case.real': 'The real satellite data (LCS-1, degrees 16-133).',
    'case.vor6_h1': 'Domain-method test: 6 Voronoi domains, plain white noise (x1).',
    'case.vor6_h2': 'Domain-method test: 6 Voronoi domains, plain white noise (x2).',
    'case.real7_h1': 'Domain-method test: the 7 domains of the real-data merging model, white noise.',
    'case.inputs': 'Input maps.',
    # ------------------------------------------------------------------ notes
    'note.null': (
        'Null test: 30 synthetic Australias with province values drawn at random and no covariate effect. '
        'The real spread of each fitted slope is 1-4x the error bar the model reports; age and "undated" '
        'come out "significant" 70% of the time. So treat regression error bars as 2-4x too small. Against '
        'the real spread, none of the real-data slopes differs from zero.'),
    'note.runcorn': (
        'A layer of uniform magnetisation produces almost no field outside it (Runcorn). Satellite data '
        'therefore see edges and contrasts much better than absolute levels, and the levels of large regions '
        'come mostly from the prior or regularisation.'),
    'note.sphere': (
        'Maps are drawn on a simple longitude-latitude grid for display only. All computations are on the '
        'sphere: equal-area cells, great-circle distances, and the field from vector spherical harmonics.'),
}
