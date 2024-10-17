from flask import Flask, request, render_template
import pandas as pd
import pickle
#import joblib


def Preprocessing(data):
    
    actual_class = data['Class']
    predict_data = data.drop(columns=['Class','Amount'])
    if 'id' in predict_data.columns:
        predict_data = predict_data.drop(columns=['id'])

    return predict_data, actual_class

app = Flask(__name__)

# Load the pre-trained model
# with open('credit_fraud.pkl', 'rb') as model_file:
# model = pickle.load(model_file)

#with open('joblib_lg_grid_model.pkl', 'rb') as model_file:
#    model = joblib.load("joblib_lg_grid_model.pkl")
model = pickle.load(open('Rand_forest.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and file.filename.endswith('.csv'):
        # Read the CSV file
        df_read = pd.read_csv(file)

        # Make predictions
        df_read, actual_class = Preprocessing(df_read)

        predictions = model.predict(df_read)

        # Add predictions to DataFrame with index
        df_index = pd.DataFrame()
        df_index['Credit Card Index'] = range(1, len(df_read)+1)
        df_index['Prediction'] = predictions

        # Convert DataFrame to a list of dictionaries for easy rendering in template
        results = df_index.to_dict('records')
        print(results)
        # Render the results template with the predictions
        return render_template('results.html', results=results)
    return 'Invalid file format', 400


if __name__ == '__main__':
    app.run(debug=True)
