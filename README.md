# NPK Prediction Model

This project trains a machine learning model to predict Nitrogen (N), Phosphorus (P), and Potassium (K) levels based on various input features from compost data.

## Project Structure

- `npk_model_training.py`: Main script for data processing and model training
- `streamlit_app.py`: Streamlit app for visualizing predictions
- `models/`: Directory containing saved model files and metrics (created by the training script)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/hannahhh05/MultiTarget_NPK.git
   cd MultiTarget_NPK
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up Google Cloud credentials:
   - Create a service account and download the JSON key
   - Set the environment variable:
     ```
     export GOOGLE_CREDENTIALS='{"type": "service_account", ...}'
     ```

## Usage

1. Run the model training script:
   ```
   python schedule_multioutput_npk.py
   ```
   This will create the `models/` directory with trained model files and metrics.

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## GitHub Actions

The project includes a GitHub Actions workflow that automatically runs the model training script. To use it:

1. Add your `GOOGLE_CREDENTIALS` as a secret in your GitHub repository settings.
2. Push changes to the repository to trigger the workflow.

## Notes

- The `models/` directory is ignored by git to avoid committing large files.
- Ensure you have the necessary permissions to access the Google Sheets data.

## Dependencies

- pandas
- numpy
- scikit-learn
- joblib
- gspread
- google-auth
- streamlit

