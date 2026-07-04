# Knicks Finals Predictor

I built this project to estimate the Knicks' championship chances using a mix of Bayesian-style Elo ratings, best-of-seven playoff simulations, opponent vulnerability analysis, and Knicks personnel actionability scoring.

The project started with a team-level Elo model. I used historical NBA game logs to estimate team strength, home-court advantage, rating uncertainty, and game-by-game Elo movement. The final Elo table gives each team a mean rating, uncertainty, and a 95% rating interval.

I then built a tournament simulator around the current simplified Finals path. I assumed the Knicks had already reached the Finals, simulated OKC vs SAS in the Western Finals, and then estimated how often the Knicks beat either opponent in a best-of-seven series.

After the baseline forecast, I analyzed why the Knicks lose games. I compared Knicks wins and losses across shooting efficiency, opponent shooting, turnovers, rebounding, free throws, assists, and other team-level features. The biggest loss indicators were lower Knicks eFG%, higher opponent eFG%, lower Knicks 3P%, higher opponent 3P%, worse rebounding margin, and weaker turnover/assist control.

I also analyzed OKC and SAS directly. For OKC, I looked at team vulnerabilities, SGA-related thresholds, bench production, and player-level loss drivers. For SAS, I focused on Wembanyama efficiency, rim impact, turnovers, opponent three-point pressure, bench production, and rebounding control.

I used those vulnerability findings to create what-if scenarios. The scenarios translate historically associated opponent-loss conditions into conservative probability gains for the Knicks. These are not causal guarantees, but they help estimate how much specific tactical improvements could matter.

Finally, I built a Knicks personnel and lineup actionability layer. I ranked Knicks players and approximate two-man and three-man combinations by how well they support the key tactical goals: pressuring SGA, surviving OKC bench minutes, creating three-point pressure against SAS, and building spacing/rebounding groups against Wembanyama.

The current pipeline produces CSVs and plots for:

- Bayesian-style Elo ratings
- Game-level Elo history
- Finals matchup simulations
- Knicks loss decomposition
- OKC and SAS vulnerability analysis
- Player-driver analysis
- Threshold loss-rate lift
- What-if scenario title probability gains
- Knicks personnel and lineup actionability rankings

The main limitation is that the lineup analysis is based on game-log co-appearances, not true possession-level lineup data. I treated those results as actionability proxies, not proof that a specific lineup causes a specific outcome.

Overall, I built this as a forecasting and basketball strategy project: first estimating who is most likely to win, then identifying why the Knicks struggle, then translating those findings into practical matchup ideas.
