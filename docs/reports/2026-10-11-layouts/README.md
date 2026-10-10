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

## Caveats

- Each synthetic check uses one noise draw.
- Locally, the null was run only for North America, Antarctica and Australia.
- Random tiles are more even in size than the polygons. That cuts against the polygons on the null, so a null closer in size could narrow the margins.

## Decisions for you

1. **Start DF-GLC (idea 2)?** The local results suggest the boundaries act through the patterns, not the levels.
2. **Run the local null for Eurasia, South America, Africa and Zealandia?** About 6 h, mostly Eurasia.
3. **Which background should local fits use from now on?** The global layout fit, or GCM-GLC.
