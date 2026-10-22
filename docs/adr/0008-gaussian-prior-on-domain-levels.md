# Domain levels in a global fit scatter about one level, with a fitted spread

When GCM is given a layout, each domain's level is drawn from a Gaussian about one global level, N(v0, tau^2). The spread tau is fitted by the evidence, and the cells scatter about their domain's level as they do about the uniform level now. The prior is Gaussian and proper, so the problem stays linear. The domain levels then add only a low-rank (Woodbury) update to the cached eigendecomposition, and a whole ladder of layouts costs about as much as one GCM run. Layouts with different numbers of domains can be compared fairly, because the prior width is fitted, not chosen, and tau -> 0 gives back the one-level GCM. A level that comes out below zero is reported, not prevented, as DF does for VIS inside a domain.

## Considered Options

- Uniform 0-4 SI·km per level, as in DF: rejected for global layouts. The levels would need sampling. The evidence would also depend on the arbitrary 0-4 width, which charges about log(4/sigma) for every extra domain, and that would weigh heavily against the ~750-province layout.
- Flat (improper) levels: rejected, because the evidence is then undefined between layouts.
