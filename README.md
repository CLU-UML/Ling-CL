# Ling-CL: Understanding NLP Models through Linguistic Curricula

The linguistic curriculum learning algorithm has three features. a) Estimating the importance of linguistic indices using a data-driven approach, b) The application of a "linguistic curriculum" to enhance the model's performance from a linguistic perspective, and c) Identifying the core set of linguistic indices needed to learn a task. This tool also evaluates the model's ability to handle different linguistic indices.

## Ling-CL
<p align="center">
<img src="https://github.com/CLU-UML/Ling-CL/assets/22674819/c34524d8-4bba-48d0-8b97-c3d442a60c1f" alt="drawing" height="300"/>
<img src="https://github.com/CLU-UML/Ling-CL/assets/22674819/30067f04-8b8a-466c-922c-4506c0dfc5b1" alt="drawing" height="300"/>
</p>

In order to apply the correlation or optimization approaches of linguistic indices importance estimation, use the following options.

`python train.py --diff_score lng_w --lng_method [opt OR corr]`

## Curriculum
![image](https://github.com/CLU-UML/Ling-CL/assets/22674819/54814a07-ae6f-4a3c-870f-d5385f36c8f6)

To apply the sigmoid, negative-sigmoid, or gaussian curricula, use the following options.

`python train.py --curr [sigmoid OR neg-simoid OR gauss]`

## Binned Balanced Accuracy
<p align="center">
<img src="https://github.com/CLU-UML/Ling-CL/assets/22674819/d99e913e-56be-402c-9fb3-69678e37d61b" alt="drawing" height="300"/>
</p>

To compute the binned balanced accuracy according to a linguistic index, you may use the function `calc_bal_acc` in [utils.py](utils/utils.py).

## Data
All datasets used are publicly available on [HF-Datasets](https://huggingface.co/datasets). The preprocessing scripts we use are available on [scripts/data](scripts/data).

To compute the linguistic indices for a dataset, scripts are provided in [scripts/tools](scripts/tools).

### Environment
Python 3.6.10
