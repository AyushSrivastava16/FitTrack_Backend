from flask import Flask
from flask_cors import CORS
from app.routes import bp

app = Flask(__name__)

# Allow CORS from React frontend
CORS(app, resources={r"/analyze/*": {"origins": "http://localhost:5173"}})

app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(debug=True)
