from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import joblib
import pandas as pd


app = Flask(__name__)

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('input', type=dict)

class prediction(Resource):
    def get(self):
        args = request.get_json()
        df = pd.DataFrame(args, index=[0])
        print(df)

        model = joblib.load('model.joblib')
        prediction = {'predicted_price': model.predict(df.values)[0]}

        return jsonify(prediction)


api.add_resource(prediction, '/prediction')

if __name__ == '__main__':
    app.run(debug=True)


# curl -X GET -H "Content-type: application/json" -d "{\"rooms\" : 3, \"floor\" : 2, \"floors_in_house\" : 5, \"floor_relation\" : 0.25, \"area\" : 62.0, \"area_living\" : 46.0, \"area_kitchen\" : 6.0}" "localhost:5000/prediction"
