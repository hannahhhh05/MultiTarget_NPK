# Timeseries_Nitrogen

## Description
This project provides a Streamlit web application for forecasting Nitrogen levels in compost using time series analysis. It fetches data from Google Sheets, processes it, and uses ARIMA models to generate forecasts.
Features

Data retrieval from Google Sheets
Time series analysis using ARIMA models
Interactive web interface built with Streamlit
Customizable forecast parameters
Visualizations of historical data and forecasts
Downloadable forecast results

## Prerequisites

* Python 3.7+
* Google Cloud Platform account with enabled Google Sheets API
* Streamlit account for deployment


### Install required packages:
* Copy pip install -r requirements.txt

### Set up Google Cloud credentials:

* Create a service account and download the JSON key file
* Rename the key file to google_credentials.json and place it in the project root (do not commit this file to GitHub)



## Configuration

Update the base_directory_id in app.py with your Google Drive folder ID containing the data sheets.
If deploying to Streamlit Cloud, add your Google credentials as a secret named GOOGLE_CREDENTIALS in the Streamlit Cloud dashboard.

## Usage
* Local Development
* Run the Streamlit app locally:
* Copy streamlit run app.py

## Streamlit Cloud Deployment

* Push your code to GitHub (ensure google_credentials.json is in .gitignore).
* Connect your GitHub repo to Streamlit Cloud.
* Add your Google credentials as a secret in the Streamlit Cloud dashboard.
* Deploy the app.

## Project Structure

* streamlit_pottasium_timeseries.py: Main Streamlit application
* schedule_pottasium_timeseries.py: Data processing and ARIMA model functions
* requirements.txt: List of Python dependencies
* .gitignore: Specifies intentionally untracked files to ignore
