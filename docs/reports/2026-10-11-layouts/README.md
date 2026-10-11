# Overnight runs, 10–11 October 2026: do the geological polygons describe crustal magnetisation?

*Plan and decisions: [experiment-global-layouts.md](../../experiment-global-layouts.md), ADRs 0007 (amendment), 0008–0010. Figures: `make_maps.py` and `make_chart.py` in this folder, run from the repository root.*

## The question

We asked whether VIS changes at the boundaries of the Hasterok et al. (2022) provinces, both in the global fit to the Gauss coefficients and in the continent fits. For each fit, the polygons were compared with **random tiles of the same number**: patches placed with no regard to geology. Grouping cells into any patches helps when VIS is smooth over a few hundred km, so the polygons only count if they beat the random tiles.

Each comparison was also run on a **null**: a smooth random truth that ignores geology. On the null the polygons should not win. Crust1.0 turned out not to be a null, because crustal thickness itself follows the provinces.

## 1. Globally, VIS follows the polygons

![](polygons_vs_random.png)

On real data the province polygons beat random tiles of the same number by **217** log units, and the province groups beat theirs by **156**. On the null they lose by 460 and 311. The global fits use the ocean-model error (ADR 0009); the earlier fits with a fitted noise scale gave the same ranking (+274 and +184).

The synthetic checks behave as they should:

| Truth | Result |
|---|---|
| One level everywhere | No layout gains more than 2 log units |
| Group levels | Groups win, and every random-tile control is at least 2480 behind |
| Smooth random field | The polygons lose to the random tiles (see above) |

The best global map gives each province its own level:

![](global_vis.png)

Compared with one level for all continents (bottom right):
- **Lower:** the Congo–Kalahari region of Africa, India, Southeast Asia and Dronning Maud Land.
- **Higher:** central North America, western Siberia, West Africa, Australia and Wilkes Land.

The province levels spread by 0.25 SI·km about a global level of 0.71 SI·km.

## 2. On the continents, the answer differs

The same test in the continent fits (DF, with GCM-GLC as the background outside each continent):

| Continent | Polygons vs random tiles | Null (random truth) |
|---|---|---|
| Eurasia | **+257** | not run |
| North America | **+127** | −55 |
| Antarctica | **+63** | −9 |
| Australia | **+34** | −23 |
| South America | +6 | not run |
| Africa | +4 | not run |
| Zealandia | −4 | not run |

The polygons carry information in Eurasia, North America, Antarctica and Australia. In South America, Africa and Zealandia, random tiles do as well.

## 3. Sharing one spread for the domain means removes most negative VIS

Each domain's mean is now drawn from one shared distribution with a fitted spread, instead of being free between 0 and 4 SI·km (ADR 0007, amendment). The share of cells below zero falls:
- Africa from 14% to **4%**;
- South America from 11% to **4%**;
- Zealandia from 3.2% to 2.5%.
- Eurasia from 2.1% to 1.6%. Its domain means that lay outside 0–4 SI·km (South India and Sri Lanka, small Aegean and Anatolian units) are now all inside.

Other continents are unchanged (0.3–1%).

![](df_afr_means.png)

In Africa, North America, Antarctica and Eurasia the fitted spread of the domain means drops to its lower bound (0.01 SI·km): **in these continents the domains hardly differ in their mean level.** Where the polygons help, they do so through the patterns inside each domain, which start afresh at the boundaries. This points to idea 2 (DF-GLC: separate patterns per domain in the global fit) as the next step, rather than more work on domain levels.

## 4. The ocean model's error is real, and confined to the oceans

The ocean-model error of ADR 0009 was confined to the oceans, and the evidence chose an amplitude of **16 nT** RMS Br. Adding it raises the global evidence by about 3300 log units over measured noise alone, but leaves the ranking of the layouts unchanged.

## 5. A better background outside each continent changes little

The local fits were run again, with the best global layout fit as the background outside each continent in place of GCM-GLC. Every continent changed only slightly:
- the mean levels fell by 0.01–0.03 SI·km (e.g. Australia 0.65 to 0.62);
- the evidence and the ring misfit barely moved.

The exception is Eurasia: its evidence rises by 60, the ring misfit falls from 27 to 26 nT, and its mean level drops from 0.60 to 0.55 SI·km.

The rise of the continent means toward GCM's level, noted in the handover, is therefore mostly not the background's doing.

## 6. DF-GLC: separate patterns per domain in the global fit (added 11 October, afternoon)

DF-GLC is DF's model fitted to the global coefficients: a fractal pattern inside each domain, independent between domains, so the pattern breaks at every domain boundary, with the ocean-model error at 16 nT. It is compared against one pattern with no breaks anywhere. Each layout's pattern shape (β, patch size) was searched separately.

| Layout | Real LCS-1 | S3: truth breaks at group edges | S5: truth with no breaks |
|---|---|---|---|
| One pattern, no breaks | **0** (β 2.5, 350 km) | −1170 | **0** |
| Groups | −55 (β 2.0, 700 km) | **0** | −628 |
| Provinces | −94 (β 2.0, 700 km) | — | — |
| Random tiles as many as the groups | −423 | −1623 | −546 |
| Random tiles as many as the provinces | −626 | — | — |

(Log evidence relative to the best layout in each column.)

- **The method works.** It finds breaks where the truth has them (S3: groups by 1170) and rejects them where it has none (S5). In S3 it also recovers the truth's strength (0.40 SI·km) and the spread of the levels (0.22 against 0.20).
- **Real VIS shows no hard breaks at province boundaries.** One pattern running across all boundaries fits best, by 55 over groups and 94 over provinces.
- **But geological boundaries are far closer to real VIS edges than random ones.** If breaks are imposed, those at group boundaries cost 368 less than breaks at random places, and those at provinces 532 less. On S5 (no edges in the truth) the groups did 83 worse than random tiles, not better.
- **Together:** VIS changes more across geological boundaries than elsewhere, but it does not start afresh at each one. That is the soft-boundary model deferred in ADR 0010: one pattern shared across boundaries, plus independent domain patterns, with the split between the two fitted by the evidence.

The DF-GLC map (one pattern) has v0 0.71 SI·km, 2.1% of cells below zero and an RMS misfit of 25.0 nT: `inversions/results/global/df_glc_lcs1_16-133_one_best_maps.png`.

## 7. Could any polygons do as well? 100 layouts that ignore geology (added 11 October, evening)

Using the polygons assumes they carry real information about magnetisation. This tests that assumption. The 724 provinces (global fit, as in section 1, ocean-model error 16 nT) were ranked against:
- **50 random tilings:** 724 Voronoi tiles each, seeds drawn at random over the continents.
- **50 spun province sets:** each continent's real polygons rotated about its centre by a random 60–300°, then trimmed to the continent. They keep the polygons' shapes and their mix of sizes; only their position relative to the geology changes.

The same 100 layouts were also fitted to the null (S5: a smooth random truth that ignores geology).

![](ensemble_evidence.png)

| | Best | Median | Worst | Beat the provinces |
|---|---|---|---|---|
| Random tilings, real data | −83 | −277 | −404 | 0 of 50 |
| Spun provinces, real data | −360 | −454 | −531 | 0 of 50 |
| Random tilings, null | +502 | +407 | +332 | 50 of 50 |
| Spun provinces, null | +44 | −58 | −138 | 5 of 50 |

(Log evidence relative to the real provinces.)

- **On real data, none of the 100 beats the provinces.** The best random tiling falls 83 log units short, and the best spun set 360 short.
- **On the null, the provinces have no advantage.** They sit among the spun sets (5 of 50 beat them), and every random tiling beats them. Random tiles are more even in size, which suits a truth that is equally smooth everywhere.
- **Position is what matters, not shape or size.** The spun sets have the same shapes and sizes as the provinces. They match the provinces on the null but lose by 360–530 on real data.

![](ensemble_maps.png)

Top: the layouts (colours only tell the domains apart). Bottom: the fitted VIS averaged over each domain. With the real provinces the averages follow known structures: the Congo low, the high cratons of West Africa and North America, and Wilkes Land. The best random and spun layouts break these up or smear them across several domains.

## Caveats

- Each synthetic check uses one noise draw, and the 100-layout test one noise draw of the null.
- Locally, the null was run only for North America, Antarctica and Australia.
- Random tiles are more even in size than the polygons. That cuts against the polygons on the null, so a null closer in size could narrow the margins. The spun province sets (section 7) remove this difference and give the same answer.

## Decisions for you

1. ~~Start DF-GLC (idea 2)?~~ Done, see section 6. Next candidate: **the soft-boundary model** (ADR 0010, deferred). It adds one hyperparameter (the split), so it costs about the same as a DF-GLC layout.
2. **Run the local null for Eurasia, South America, Africa and Zealandia?** About 6 h, mostly Eurasia.
3. **Which background should local fits use from now on?** The global layout fit, or GCM-GLC.
