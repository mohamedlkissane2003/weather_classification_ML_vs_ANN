# Weather Classification: ML vs ANN

This is a simple Python project that collects weather data from Open-Meteo for several Moroccan cities, cleans and prepares the data, and then compares two classification approaches:

- Logistic Regression as a traditional machine learning baseline
- Artificial Neural Network as a neural network approach

At the end, the script displays each model's accuracy, confusion matrix, and the ANN training curves.

## Project Idea

The script uses historical/recent weather data for each city and converts `weathercode` into 3 main weather classes:

- `0` clear or mostly clear weather
- `1` partly cloudy or variable weather
- `2` rainy or worse weather

The following features are then used for prediction:

- Temperature
- Humidity
- Wind speed
- Surface pressure
- Cloud cover
- Hour
- Month

## Requirements

- Python 3.9 or later
- Internet connection to fetch data from Open-Meteo

### Required Packages

- `requests`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `tensorflow`

## Installation

Install the dependencies manually with:

```bash
pip install requests pandas numpy seaborn matplotlib scipy scikit-learn tensorflow
```

## How to Run

Run the main script:

```bash
python weather_classification_ml_vs_ann.py
```

## Files Generated Automatically

When the script runs, it saves several CSV files in the same folder:

- `weather_raw.csv`
- `weather_data_cleaned.csv`
- `weather_model_dataset.csv`
- `weather_train.csv`
- `weather_test.csv`

## Expected Output

During execution, the script will display:

- Data size after collection and cleaning
- Target class distribution
- Logistic Regression results
- ANN results
- Confusion matrix for each model
- Loss and accuracy curves for the neural network

## Notes

- The project uses `matplotlib.pyplot.show()`, so plots may open in separate windows when running locally.
- If the API request fails, data collection will not complete.

## License

No license has been specified for this project yet.