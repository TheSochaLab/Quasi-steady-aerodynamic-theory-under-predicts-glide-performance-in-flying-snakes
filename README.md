# Quasi-steady aerodynamic theory under-predicts glide performance1 in flying snakes

This repository contains code and figures for the manuscript in preparation,
"Quasi-steady aerodynamic theory under-predicts glide performance1 in flying snakes".

## Code structure

- `Code` contains Python scripts to process the raw data and produce figures.
- `Figures` contains the exported figures.

The flying snake kinematics data used in this analysis was described in Yeaton et. al. (2020) [[1](#ref1)]. Please refer to that works code repository [data folder](https://github.com/TheSochaLab/Undulation-enables-gliding-in-flying-snakes/tree/master/Experiments/Data) and [data repository](https://drive.google.com/drive/folders/1FpSBUD1XY3guuWjGUE5V7dqluNkoXyKy).

## Exporting dependencies

```bash
conda activate gaf
conda env export --from-history > environment.yml
conda env export  # copy -pip section to environment.yml
```

## References

[[1](#ref1)] Yeaton IJ, Ross SD, Baumgardner GA, Socha JJ. 2020. Undulation enables gliding in flying snakes. Nature Physics. 16(9):974-982.
