# Federated Learning for Plant Disease Detection

Code and experiment logs for the manuscript **"A Comparative Study of Federated Learning Algorithms for Plant Disease Detection under Non-IID Data"** by Jaspal Kaur Saini and Lovepreet Sharma (Dr. B. R. Ambedkar National Institute of Technology, Jalandhar).

The study compares four federated learning (FL) algorithms, **FedAvg**, **FedProx**, **FedMA** and **FedOpt**, for image-based plant disease classification on five public datasets, with data partitioned across clients in a non-IID way to mimic farms in different locations.

> **Status:** manuscript under review.

## Repository structure

```
notebooks/<dataset>/   Jupyter notebooks, one per experiment (as run on Kaggle)
logs/<dataset>/        full console output of each run (per-round metrics, final results)
results/<dataset>/     per-round metric CSVs, training curves and confusion matrices
```

Each notebook contains the complete pipeline: data loading, Dirichlet non-IID partitioning, local training, server aggregation and evaluation. Notebooks were executed as Kaggle batch jobs, so the printed output of each run is kept in the matching file under `logs/`. The exception is `fedma_cotton.ipynb`, which was run interactively and keeps its output inside the notebook.

## Experimental setup

| Setting | Value |
|---|---|
| Model | ResNet18 (ImageNet-pretrained), final layer resized to the number of classes |
| Clients | 5 |
| Data partition | Dirichlet, α = 0.5 (non-IID) |
| Local epochs per round | 5 |
| Batch size | 32 |
| Local learning rate | 0.001 (0.0001 for FedOpt on Potato, Tomato and PlantVillage) |
| FedProx proximal term | μ = 0.01 |
| FedOpt server optimizer | Adam, learning rate 0.001 |
| Communication rounds | 10 (FedOpt: 50) |
| Hardware | Kaggle, NVIDIA Tesla T4 GPU |
| Software | Python 3.12, PyTorch 2.10 (CUDA 12.8) |

Metrics are the global model's accuracy, and macro-averaged precision, recall and F1-score, on a held-out test split.

## Results

FedAvg, FedProx and FedMA are reported at **communication round 10**; they converge within 10 rounds, and training beyond that was not needed. For example, FedAvg on Potato reached 0.9861 accuracy at round 10 and 0.9838 at round 50. FedOpt did not give good results at 10 rounds (accuracy 0.44–0.82), so it is reported at **round 50** (on Wheat its training diverged and the round-10 value is shown).

| Dataset | Algorithm | Notebook | Round | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Cotton | FedAvg | [`fedavg_fedprox_cotton.ipynb`](notebooks/cotton/fedavg_fedprox_cotton.ipynb) | 10 | 0.9927 | 0.9928 | 0.9927 | 0.9927 |
| Cotton | FedProx | [`fedavg_fedprox_cotton.ipynb`](notebooks/cotton/fedavg_fedprox_cotton.ipynb) | 10 | 0.9948 | 0.9948 | 0.9948 | 0.9948 |
| Cotton | FedMA | [`fedma_cotton.ipynb`](notebooks/cotton/fedma_cotton.ipynb) | 10 | 0.9634 | 0.9647 | 0.9627 | 0.9628 |
| Cotton | FedOpt | [`fedopt_cotton.ipynb`](notebooks/cotton/fedopt_cotton.ipynb) | 50 | 0.9079 | 0.5036 | 0.5074 | 0.5051 |
| Wheat | FedAvg | [`fedavg_wheat.ipynb`](notebooks/wheat/fedavg_wheat.ipynb) | 10 | 0.8466 | 0.8356 | 0.8270 | 0.8293 |
| Wheat | FedProx | [`fedprox_wheat.ipynb`](notebooks/wheat/fedprox_wheat.ipynb) | 10 | 0.8437 | 0.8390 | 0.8437 | 0.8387 |
| Wheat | FedMA | [`fedma_wheat.ipynb`](notebooks/wheat/fedma_wheat.ipynb) | 10 | 0.8473 | 0.8375 | 0.8261 | 0.8288 |
| Wheat | FedOpt | [`fedopt_wheat.ipynb`](notebooks/wheat/fedopt_wheat.ipynb) | 10 | 0.0576 | 0.0033 | 0.0576 | 0.0063 |
| Potato | FedAvg | [`fedavg_fedprox_fedma_potato.ipynb`](notebooks/potato/fedavg_fedprox_fedma_potato.ipynb) | 10 | 0.9861 | 0.9865 | 0.9861 | 0.9857 |
| Potato | FedProx | [`fedavg_fedprox_fedma_potato.ipynb`](notebooks/potato/fedavg_fedprox_fedma_potato.ipynb) | 10 | 0.9861 | 0.9859 | 0.9861 | 0.9858 |
| Potato | FedMA | [`fedavg_fedprox_fedma_potato.ipynb`](notebooks/potato/fedavg_fedprox_fedma_potato.ipynb) | 10 | 0.9838 | 0.9843 | 0.9838 | 0.9831 |
| Potato | FedOpt | [`fedopt_potato.ipynb`](notebooks/potato/fedopt_potato.ipynb) | 50 | 0.9420 | 0.9459 | 0.9420 | 0.9335 |
| Tomato | FedAvg | [`fedavg_tomato.ipynb`](notebooks/tomato/fedavg_tomato.ipynb) | 10 | 0.9927 | 0.9928 | 0.9927 | 0.9927 |
| Tomato | FedProx | [`fedprox_tomato.ipynb`](notebooks/tomato/fedprox_tomato.ipynb) | 10 | 0.9823 | 0.9825 | 0.9823 | 0.9822 |
| Tomato | FedMA | [`fedma_tomato.ipynb`](notebooks/tomato/fedma_tomato.ipynb) | 10 | 0.9832 | 0.9833 | 0.9832 | 0.9831 |
| Tomato | FedOpt | [`fedopt_tomato.ipynb`](notebooks/tomato/fedopt_tomato.ipynb) | 50 | 0.8425 | 0.8673 | 0.8425 | 0.8142 |
| PlantVillage | FedAvg | [`fedavg_plantvillage.ipynb`](notebooks/plantvillage/fedavg_plantvillage.ipynb) | 10 | 0.9913 | 0.9891 | 0.9863 | 0.9876 |
| PlantVillage | FedProx | [`fedprox_plantvillage.ipynb`](notebooks/plantvillage/fedprox_plantvillage.ipynb) | 10 | 0.9623 | 0.9578 | 0.9516 | 0.9538 |
| PlantVillage | FedMA | [`fedma_plantvillage.ipynb`](notebooks/plantvillage/fedma_plantvillage.ipynb) | 10 | 0.9960 | 0.9950 | 0.9924 | 0.9936 |
| PlantVillage | FedOpt | [`fedopt_plantvillage.ipynb`](notebooks/plantvillage/fedopt_plantvillage.ipynb) | 50 | 0.9506 | 0.9563 | 0.9506 | 0.9503 |

Precision, recall and F1 are macro-averaged. Every value can be checked against the corresponding file in `logs/` (and `results/` for per-round CSVs).

### Notes on individual runs

- **FedOpt on Cotton** loaded 14 labels instead of 6, because duplicate folders in the dataset (e.g. `Aphids` and `Aphids edited`) were read as separate classes. Its macro-averaged precision, recall and F1 are therefore not comparable with the other Cotton runs.
- **FedProx on PlantVillage** used ResNet50 with 10 local epochs and reports metrics on a validation split.
- **FedOpt on Wheat** diverged early in training.
- **`fedma_tomato.ipynb`** also contains exploratory SCAFFOLD and FedOpt runs. Only its FedMA result is reported; FedOpt on Tomato is taken from `fedopt_tomato.ipynb`.
- **`fedprox_wheat.ipynb`** also contains a SCAFFOLD section, which stopped with a device-mismatch error after the FedProx run had completed. Only the FedProx result is reported.
- SCAFFOLD was explored during the project but is not part of the reported comparison.

## Datasets

The datasets are not included in this repository; they are available on Kaggle.

| Dataset | Images | Classes | Source |
|---|---|---|---|
| Cotton Plant Disease | 4,788 | 6 | [dhamur/cotton-plant-disease](https://www.kaggle.com/datasets/dhamur/cotton-plant-disease) |
| Wheat Plant Diseases | 13,104 | 15 | [kushagra3204/wheat-plant-diseases](https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases) |
| Potato Plant Diseases | 2,152 | 3 | [hafiznouman786/potato-plant-diseases-data](https://www.kaggle.com/datasets/hafiznouman786/potato-plant-diseases-data) |
| Tomato Leaf Disease | 10,000 | 10 | [kaustubhb999/tomatoleaf](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) |
| PlantVillage | 54,305 | 38 | [abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |

Image and class counts are those loaded by the notebooks.

## Reproducing the experiments

**On Kaggle (recommended):** create a notebook, import the `.ipynb` file, add the matching dataset from the table above, set the accelerator to **GPU T4**, and run all cells. Paths in the notebooks assume Kaggle's `/kaggle/input/...` layout.

**Locally:** install the requirements and change the dataset path at the top of the notebook to your local copy.

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended; on CPU a single round on the larger datasets can take several hours.

## Citation

If you use this code, please cite the manuscript (full reference to be added on publication):

```bibtex
@article{saini_sharma_fl_plant_disease,
  author  = {Saini, Jaspal Kaur and Sharma, Lovepreet},
  title   = {A Comparative Study of Federated Learning Algorithms for Plant Disease Detection under Non-IID Data},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The datasets are subject to their own licenses on Kaggle.
