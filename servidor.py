from flask import Flask, request, jsonify
import pandas as pd
import pickle
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_breast_cancer():
    data = request.json
    with open('checkpoints/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('checkpoints/svm.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('checkpoints/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
 
    df = pd.DataFrame(data)
    print(df)
    feature_columns = ['perimeter_worst','perimeter_se','radius_worst','smoothness_mean','texture_mean','fractal_dimension_mean',
                       'concave points_se','concave points_worst','texture_worst','radius_se','concavity_worst','area_mean',
                       'smoothness_worst','smoothness_se','compactness_worst','area_se','concavity_mean','symmetry_worst',
                       'radius_mean','concavity_se','fractal_dimension_se','perimeter_mean','concave points_mean',
                       'compactness_se','compactness_mean','symmetry_se','fractal_dimension_worst','symmetry_mean','area_worst',
                       'texture_se']
    df = df[feature_columns]
    df[feature_columns] = scaler.transform(df)
    pca_df = pca.transform(df[feature_columns])
    feature_columns = [f'PCA{i+1}' for i in range(15)]
    pca_df = pd.DataFrame(data=pca_df, columns=feature_columns)
    predict = model.predict(pca_df)
    response ={'cancer_type': 'Maligno' if predict[0] == 0 else 'Benigno'}
    return jsonify(response)  

if __name__ == '__main__':
    app.run(debug=True,port=8080)
