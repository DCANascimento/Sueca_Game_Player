# Sueca_Game_Player



## Dataset/:

- Contains the data gathered from all the games

## Deliverables/:

- Contains all the 3 Deliverables

## Models/:

- Contains Each of the three models used:

    - Multiclass Logistic Regression
    - Gradient Boosting
    - XGBoost

- Also contains a basic file for comparing their results and evaluating basic metrics

## Presentation/:

- Presentation shown in class


## Como correr

1. Criar um venv

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Processamento de dados (não é necessário, porque já temos um ficheiro com os dados antes e depois do seu processamento)

python3 Dataset/data-gathering/statistic-analysis/data-cleaner.py

3. Correr comparação entre modelos

cd Models
python3 model_comparison.py


**Nenhuma mudança foi feita nem ao powerpoint nem ao código desde a apresentação do dia 29/5
