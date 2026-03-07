# usecase
This repo contains modules showcasing my data science technical capabilities

## Custom Regression Pipeline (Any Dataset)

Use the reusable CLI entrypoint to train a regression pipeline on your own CSV:

```bash
python -m src.regression.run_custom --dataset data/your_data.csv --target your_target
```

### Common examples

Train on a generic dataset (disable merc-specific transforms):

```bash
python -m src.regression.run_custom \
	--dataset data/your_data.csv \
	--target sale_price \
	--disable-merc-transforms \
	--no-mlflow
```

Train with explicit schema:

```bash
python -m src.regression.run_custom \
	--dataset data/your_data.csv \
	--target sale_price \
	--numeric-cols area,age,rooms \
	--categorical-cols city,segment \
	--log-cols area,age
```

Save model to a custom location:

```bash
python -m src.regression.run_custom \
	--dataset data/your_data.csv \
	--target sale_price \
	--model-path artifacts/models/custom_reg_pipe.joblib
```

### Predict with saved pipeline

Python example:

```python
from src.regression.predict import predict_from_csv

result = predict_from_csv(
		input_csv_path="data/new_samples.csv",
		output_csv_path="artifacts/predictions/new_samples_pred.csv",
		model_path="artifacts/models/custom_reg_pipe.joblib",
)

print(result.output.head())
```
