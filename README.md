[![Open Source Love](https://firstcontributions.github.io/open-source-badges/badges/open-source-v2/open-source.svg)](https://github.com/firstcontributions/open-source-badges)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

This repository consists of the work done as a student developer and researcher, during the Google Summer of Code (GSoC) 2021 program. The project, "Decoding quantum states through nuclear magnetic resonance" is a part of the NMR suburbanization under the Machine Learning for Science (ML4SCI) umbrella organization.

## Description

At low temperatures, many materials transition into an electronic phase which cannot be classified as a simple metal or insulator. So-called quantum phases of matter, like superconductors and spin liquids, are hard to study due to their fragile nature, making nonintrusive and indirect measurements important. We intend to explore the connection between electronic phases and nuclei in these materials via simulations of nuclear magnetic resonance (NMR). By using external magnetic pulses, the nuclear spins can be controlled, and their time-evolution (time-dependent magnetization) studied.

## Task ideas

- To implement a classification model that determines the type of electronic interaction based on only the time-dependent curve. How sensitive is this classification to noise?
- Develop, Benchmark and program a neural network to predict the strength, range, and dissipation parameters of a given magnetization curve.
- Develop an algorithm which optimizes an applied pulse sequence to best estimate a specific physical parameter from a given material.
