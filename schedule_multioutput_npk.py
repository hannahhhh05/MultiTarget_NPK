import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import gspread
from google.oauth2.service_account import Credentials # type: ignore
from googleapiclient.discovery import build # type: ignore
from googleapiclient.errors import HttpError
import json
import os

def create_gspread_client(credentials_path):
    return gspread.authorize(credentials_path)

def authenticate_drive(credentials_path):
    return build("drive", "v3", credentials=credentials_path)

def get_sheets_in_folder(drive_service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'"
    results = drive_service.files().list(q=query, pageSize=1000, fields="files(id, name)").execute()
    return results.get('files', [])

def read_and_concat_sheets(client, file_id, header_row=1):
    spreadsheet = client.open_by_key(file_id)
    all_sheets_data = []
    for sheet in spreadsheet.worksheets():
        sheet_data = pd.DataFrame(sheet.get_all_values())
        # Check if the first row has empty values
        if any(pd.isna(sheet_data.iloc[header_row - 1])):
            sheet_data.columns = sheet_data.iloc[header_row]
        else:
            sheet_data.columns = sheet_data.iloc[header_row - 1]
        sheet_data.reset_index(drop=True, inplace=True)  # Reset index
        final_columns = [
            "Timestamp",
            "Number of Worms (non-counted)",
            "Phosphorous01",
            "Phosphorous02",
            "Nitrogen01",
            "Nitrogen02",
            "Potassium01",
            "Potassium02",
            "Light Intensity",
            "Temp01",
            "Hum01",
            "Heat01",
            "SoilM01",
            "SoilM02",
            "Buzzer",
            "pH Rod 1",
            "pH Rod 2",
        ]
        sheet_data = sheet_data.loc[:, ~sheet_data.columns.duplicated()]
        sheet_data = sheet_data.reindex(columns=final_columns, fill_value=None)
        print(sheet_data.shape)
        # sheet_name = sheet.title
        # print("Sheet Name:", sheet_name)
        all_sheets_data.append(sheet_data)
    # Concatenate all sheet data into a single DataFrame
    return pd.concat(all_sheets_data, ignore_index=True)

def load_and_concat_all_sheets_in_centers(base_directory_id, credentials_path):
    client = create_gspread_client(credentials_path)
    drive_service = authenticate_drive(credentials_path)

    compost = []
    center_folders = (
        drive_service.files()
        .list(
            q=f"'{base_directory_id}' in parents and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)",
        )
        .execute()
        .get("files", [])
    )

    for center_folder in center_folders:
        sheet_files = get_sheets_in_folder(drive_service, center_folder["id"])
        for sheet_file in sheet_files:
            sheet_data = read_and_concat_sheets(client, sheet_file["id"])
            sheet_data["Location"] = center_folder["name"].split("_")[1]
            compost.append(sheet_data)

    return pd.concat(compost, ignore_index=True)

def preprocess_data(data):
    data = data.dropna(subset=["Timestamp"])
    
    data = data[~data["Timestamp"].str.contains("Unit|Timestamp")]
        
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    numeric_columns = [
        'Number of Worms (non-counted)', 'Phosphorous01', 'Phosphorous02', 'Nitrogen01', 
        'Nitrogen02', 'Potassium01', 'Potassium02', 'Light Intensity', 'Temp01', 
        'Hum01', 'Heat01', 'SoilM01', 'SoilM02', 'Buzzer', 'pH Rod 1', 'pH Rod 2'
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    threshold = len(data) * 0.40
    
    data = data.dropna(axis=1, thresh=threshold)
    print("Columns after dropping those with >60% null values:")
    print(data.columns)

    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    data['Phosphorous'] = data[['Phosphorous01', 'Phosphorous02']].mean(axis=1)
    data['Nitrogen'] = data[['Nitrogen01', 'Nitrogen02']].mean(axis=1)
    data['Potassium'] = data[['Potassium01', 'Potassium02']].mean(axis=1)

    data = data.drop(['Phosphorous01', 'Phosphorous02', 'Nitrogen01', 'Nitrogen02', 'Potassium01', 'Potassium02'], axis=1)
    
    le = LabelEncoder()
    data['Location'] = le.fit_transform(data['Location'])
    
    # Separate features for scaling (exclude 'Location' and target variables)
    features_to_scale = [col for col in data.columns if col not in ['Location', 'Timestamp', 'Phosphorous', 'Nitrogen', 'Potassium']]
    
    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    
    print("\nDataset after preprocessing:")
    print(data.head())
    print("\nDataset info:")
    print(data.info())
    
    return data, scaler, le

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'estimator__n_estimators': [50, 100, 150],
        'estimator__max_depth': [10, 20, None]
    }
    
    grid_search = GridSearchCV(MultiOutputRegressor(RandomForestRegressor(random_state=42)), 
                               param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    
    return best_model, mse, r2

def get_credentials():
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS environment variable is not set")

    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError:
        raise ValueError("GOOGLE_CREDENTIALS is not a valid JSON string")

    required_fields = ['client_email', 'token_uri', 'private_key']
    for field in required_fields:
        if field not in creds_dict:
            raise ValueError(f"GOOGLE_CREDENTIALS is missing required field: {field}")

    return Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
    )

def main():
    base_directory_id = '1uH8e33HJQG4v8BmCZ_EJYmGH-VmWI547'
    credentials_path = get_credentials()
    try:
        raw_data = load_and_concat_all_sheets_in_centers(base_directory_id, credentials_path)
        processed_data, scaler, le = preprocess_data(raw_data)
        
        X = processed_data.drop(columns=['Timestamp', 'Phosphorous', 'Nitrogen', 'Potassium'])
        y = processed_data[['Phosphorous', 'Nitrogen', 'Potassium']]
        
        X['Location'] = X['Location'].astype('category')
        
        best_model, mse, r2 = train_model(X, y)
        
        # Create a 'models' directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model, scaler, and label encoder in the 'models' directory
        joblib.dump(best_model, 'models/best_npk_model.pkl')
        joblib.dump(scaler, 'models/npk_scaler.pkl')
        joblib.dump(le, 'models/location_encoder.pkl')
        
        feature_info = {
            'features': X.columns.tolist(),
            'categorical_features': ['Location'],
            'numeric_features': X.select_dtypes(include=[np.number]).columns.tolist(),
            'target_variables': y.columns.tolist()
        }
        with open('models/feature_info.json', 'w') as f:
            json.dump(feature_info, f)
        
        print(f"Best MSE: {mse}")
        print(f"Best R2: {r2}")
        print(f"Features used in training: {X.columns.tolist()}")
        
        with open('models/model_metrics.txt', 'w') as f:
            f.write(f"MSE: {mse}\n")
            f.write(f"R2: {r2}\n")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()