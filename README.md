# Deep (Predictive) Discounted Counterfactual Regret Minimization

> Deep (Predictive) Discounted Counterfactual Regret Minimization <br>
> Hang Xu, Kai Li<sup>\#</sup>, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng <br>
> IJCAI 2024 (Oral)

## Install DeepPDCFR

Install miniconda3 from [the official website](https://docs.conda.io/en/latest/miniconda.html) and run the following script:

```bash
bash scripts/install.sh
```

## Train DeepPDCFR

Run the following script to assess the performance of model-free neural algorithms on testing games. The results are saved in the folder `logs`.
```bash
conda activate DeepPDCFR
python scripts/run.py with configs/{algo_name}.yaml game_name={game_name} seed={seed} --force
```
`algo_name` is the algorithm name chosen from `NFSP, QPG, RPG, OSDeepCFR, VRDeepDCFRPlus, VRDeepPDCFRPlus`.

`game_name` is the testing game name chosen from `KuhnPoker, LeducPoker, LiarsDice5, LiarsDice6, GoofSpielImp5, GoofSpielImp6, Battleship_22_3, Battleship_32_3, FHP`

`seed` is the random seed chosen from `0, 1, 2, 3`.
    
