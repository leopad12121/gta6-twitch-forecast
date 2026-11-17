# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/leopad12121/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/leopad12121/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people will be watching Grand Theft Auto VI live on
Twitch in each hour of its release day, 2026-11-19 (UTC), given only the
first hour's count — as five quantiles an hour (5%, 25%, 50%, 75%, 95%),
scored by pinball loss on the log scale.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, no network, no
randomness. It reads every `<id>.in` in the input directory (one case:
the GTA VI category's Twitch count up to the end of hour 1 of release
day, plus 38 past launches as comparables) and writes `<id>.out` with
23 rows, hours 2..24. Two runs on the same case write the same bytes
(verified); one case takes ~0.07 s (cap 8 s).

## How it works

For each hour h = 2..24 the model forecasts the log-ratio
`r_h = ln(viewers_h / viewers_1)` and reports
`q_tau(h) = p1 * exp(mu_h + z_tau * sigma_h)`, blending two views of
the 38 comparable launch days shipped in the case file:

- **Rolling-midnight regression (weight 0.7).** Per-hour OLS of the
  log-ratio on `ln(p1)` over the 12 comparables with a rolling
  midnight-local unlock — the unlock GTA VI itself has (New Zealand
  first, 2026-11-18T11:00Z). Big first hours mean the world was
  already watching, so the curve stays flat; small first hours mean
  the peak is still ahead. Across those launches, `log(h1)` and
  `log(peak/h1)` correlate at -0.94.
- **Clean rolling shape (weight 0.3).** Per-hour median log-ratio
  over the 9 rolling launches without a PC-simultaneous unlock or an
  early-access window (ELDEN RING, Cyberpunk 2077 and Hogwarts Legacy
  excluded — their hour-1 prints already hold the whole world).
  Median ratio rises from ~0.9 in the small hours to ~2.5 at hours
  19–21 UTC, when Europe's evening meets the US afternoon.

Spread is widened by 1.15× to the honest 90% coverage the
leave-one-out check shows. If a case carries too few usable
comparables the program falls back to coefficients fitted offline from
the same 38 launches (embedded in `main.py`).

## Validation

Leave-one-out over the comparables, scored with the contest's pinball
loss on the log scale (`python3 tools/validate.py` reproduces this):

| model | rolling 12 | all 37 |
|---|---|---|
| house baseline (persistence) | 0.357 | 0.930 |
| this model | **0.246** | 0.672 |

The house's own practice figure is 0.387, so this beats the bar to
beat on the launches that resemble the final. Implied launch-day peaks
for the kit's practice cases: God of War Ragnarök ~491k (published
peak ~478k), Spider-Man 2 ~280k (published ~257k), Red Dead
Redemption 2 ~276k, Marvel's Wolverine ~96k.

Why the first hour is small and the peak is late: at 00:00 UTC on
release day only New Zealand, Australia, Japan and Korea have
unlocked, plus the first hour of continental Europe; the US East
unlocks at 05:00 UTC, the US West at 08:00 UTC. For scale: the GTA VI
category drew 1,418,094 viewers in one hour on 2026-08-27 (its
extended-look premiere), while the biggest launch-day category peaks
in the comparables are ~1.07M (Cyberpunk 2077) and ~0.91M (ELDEN
RING); the highest game-category peak ever on Twitch is League of
Legends at 3.11M (Worlds final, 2022).

## Sources

- Kit comparables: SullyGnome hourly Twitch category averages,
  quoted with attribution in the case files.
- GTA VI release 19 Nov 2026, midnight-local unlock: Rockstar /
  PlayStation support page.
- Trailer figures: Guinness (Trailer 1, 90.4M views in 24h);
  Rockstar via trade press (Trailer 2, >475M views in 24h).
- Launch-day peaks: Streams Charts (Elden Ring ~910k), Final
  Weapon / SullyGnome (God of War Ragnarök ~478k), TwitchStats
  (Spider-Man 2 ~257k), Streams Charts hall of fame (LoL 3.11M).

---

<sub>Entered on [make.fun](https://make.fun/?ref=d5a50dzy&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
