# nbme-score-clinical-patient-notes
Kaggle competition nbme-score-clinical-patient-notes

### Download data for the competitions
competitions download nbme-score-clinical-patient-notes

### Training with a GPU

You may want to consider to reduce the total power consumption, and thereby reduce the vRAM may temp. To find the ideal configuration, observe your vRAM under heavy GPU load. Tooling on Linux is not good for doing so. I suggest you use windows HWinfo64

```bash
sudo nvidia-smi -i 0 -pl 230
watch -n 1 nvidia-smi
```

### Use docker kaggle container

In VS Code: Ctrl + Shift + P: Remote-Containers: Rebuild and Reopen in Container
Spin up jupyter notebook: from within the container ```bash jupyter notebook --allow-root```
Spin up tensorboard: from within the container ```bash tensorboard --logdir data/working/ --bind_all```
