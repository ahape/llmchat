# Updating `models.csv`

Get the updated list of models

```sh
curl -L -o models.raw.json https://openrouter.ai/models
```

Run the script for converting it to CSV

```sh
./convert_raw_models_to_csv.py
```

Commit the changes.
