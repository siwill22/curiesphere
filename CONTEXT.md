# curiesphere

Forward models of the lithospheric magnetic field from crustal magnetisation, and inversions that estimate crustal magnetic properties from satellite field models.

## Language

### What is estimated

**VIS**:
Vertically integrated susceptibility of a column of crust, in SI·km. The quantity the inversions estimate for induced sources.
_Avoid_: magnetic thickness, depth-integrated susceptibility, κ

**VIM**:
Vertically integrated magnetisation, in A: the vector the forward model takes as input. For induced sources it is derived from VIS and the inducing field, never estimated directly.
_Avoid_: magnetisation (when the vertically integrated quantity is meant)

**RVIM**:
The remanent part of VIM, from the ocean remanence models.
_Avoid_: remanent magnetisation (when the vertically integrated quantity is meant)

**Source strength**:
The scalar a parameter multiplies onto a known magnetisation direction and depth distribution. VIS is the source strength of induced sources; a signed RVIM amplitude would be that of remanent sources.
_Avoid_: amplitude, weight

### Where sources live

**Grid node**:
A node of the 0.1° global grid (the ocean-age grid). It carries geometry only: top of basement, inducing field, ocean age or none. Never a parameter.
_Avoid_: pixel, point (points are where the field is evaluated)

**Cell**:
A HEALPix pixel (NSIDE 64, about 100 km, or a finer NSIDE) and the grid nodes inside it. The smallest unit a parameter can cover.
_Avoid_: pixel, block

**Domain**:
A group of cells sharing one source strength, whether fixed in advance (polygons) or formed from the data.
_Avoid_: unit (except for a labelled-Voronoi unit of several pieces), province, region (when one group of cells is meant)

**Parameterisation**:
A grouping of cells into parameters: each parameter is a fixed combination of cells (one cell, a domain, a coarse cell, a covariate). Its columns are the cell columns combined accordingly.
_Avoid_: basis (when the grouping, not the cell columns, is meant)

**Layer**:
A slab of crust between a top surface and a base surface (each a constant depth or a grid, such as top of basement, a Moho or a Curie-depth model), with VIS spread uniformly between them.
_Avoid_: magnetic layer (when the surfaces are not specified)

**Direction model**:
The rule that gives the magnetisation direction at each grid node: induced (along the inducing field, IGRF-13 by default) or, later, remanent (from palaeolatitude and declination on the age grid).
_Avoid_: field model (that is the data)

**Source geometry**:
The layer or layers in which the sources sit. It fixes where the magnetisation is, not how much; it is chosen and compared, never estimated. The default is one layer from top of basement to 30 km below the ellipsoid.
_Avoid_: depth model, magnetic thickness

### Setting up an inversion

**Target area**:
The area whose VIS an inversion estimates (e.g. Australia, or all continents). Its cells are always free; no prior VIS model (such as Hemant & Maus) is used inside it.
_Avoid_: region (for the area itself), extent, study area

**Background**:
Sources outside the target area held fixed at a model: the ocean model, and optionally Hemant & Maus (+ slabs) on continental crust outside the target area.
_Avoid_: prior model, reference model

**Outer cells**:
Free, damped cells on continental crust outside the target area, used instead of a Hemant & Maus background.
_Avoid_: background (they are estimated, not fixed)

**Local fit**:
An inversion fitted to Br at data points within a buffer of the target area, with everything outside the target area in the background.
_Avoid_: regional inversion

**Global fit**:
An inversion fitted to the Gauss coefficients of a field model, with outer cells on all other continental crust and only the ocean model in the background.
_Avoid_: whole-earth inversion

**Data points**:
Points (HEALPix, about 55 km) where Br is evaluated for a local fit.
_Avoid_: observations (they are evaluations of a field model)

**Synthetic data**:
Data computed from a chosen truth (plus the background and a noise realisation drawn from the error model) in place of a real field model, so that any inversion can be checked against a known answer.
_Avoid_: test data, fake data

**Truth**:
The source strengths used to compute synthetic data; stored with the results so estimates can be compared with it.
_Avoid_: true model (when the background is meant)

### Errors

**Noise**:
The part of the data error that comes from the field model itself, measured by its disagreement with an independent field model (LCS-1 against MF7 or MLI_SHA_2E).
_Avoid_: data error (when only this part is meant)

**Model error**:
The part of the field that a given parameterisation cannot reproduce, such as variation inside a domain. It belongs to a fitted model, not to the data.
_Avoid_: noise, misfit

**Error model**:
The combination of noise and model error used to weight the data: each a per-degree spectrum with an amplitude. Each amplitude is either fixed (from field-model differences or a reference fit) or fitted with the model; every result records which.
_Avoid_: covariance (for the whole construction), weighting

### Methods

**GCM** (global cell map):
VIS per cell on all continental crust from a global fit, damped toward one fitted uniform level, with the damping chosen by the evidence or the discrepancy principle.

**CM** (cell map):
VIS per cell in a target area with GMRF, L2 or TV regularisation.

**PD** (polygon domains):
One VIS per domain fixed in advance from geological polygons (Hasterok et al. 2022).

**FD** (free domains):
Domains formed from the data: FD-merge (agglomerative merging), FD-vor (Voronoi), FD-lvor (labelled Voronoi), FD-part (cell partition).

**CR** (covariate regression):
VIS modelled as a combination of covariates (crustal thickness, heat flow, province class…), with hypotheses scored by the evidence.

**ORP23**:
The method of Olsen, Ravat & Purucker (2023): point dipoles at 70 km, damped toward an a-priori thickness model. Referred to, not implemented.

A method name takes the target-area tag as a suffix: AUS (Australia), NAM (North America), GLC (all continental crust), e.g. PD-AUS, GCM-GLC.
_Avoid_: "the global run", "the regional inversion" (name the method and target area)

**Column build**:
How cell columns are computed: by transform (one global forward transform per cell) or by node sum (a sum over the cell's own grid nodes). The two agree to rounding; the choice affects speed only and never names a method.
