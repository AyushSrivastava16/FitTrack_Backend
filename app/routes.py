
from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64
from .detectors import bicep_curl

bp = Blueprint('api', __name__)

def decode_image(base64_string):
    img_data = base64.b64decode(base64_string.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@bp.route('/analyze/bicep_curl', methods=['POST'])
def analyze_bicep_curl():
    data = request.get_json()
    image = decode_image(data['image'])
    result = bicep_curl.analyze_bicep_curl(image)
    return jsonify(result)



from .detectors import pushup  # ⬅️ Add this at the top with the other imports

@bp.route('/analyze/pushup', methods=['POST'])
def analyze_pushup():
    data = request.get_json()
    image = decode_image(data['image'])
    result = pushup.analyze_pushup(image)
    return jsonify(result)




from .detectors import wall_pushup


@bp.route('/analyze/wall_pushup', methods=['POST'])
def analyze_wall_pushup():
    data = request.get_json()
    image = decode_image(data['image'])
    result = wall_pushup.analyze_wall_pushup(image)
    return jsonify(result)




from .detectors import jumping_jacks



@bp.route('/analyze/jumping_jacks', methods=['POST'])
def analyze_jumping_jacks():
    data = request.get_json()
    image = decode_image(data['image'])
    result = jumping_jacks.analyze_jumping_jacks(image)
    return jsonify(result)





from .detectors import high_knee


@bp.route('/analyze/high_knee', methods=['POST'])
def analyze_high_knee():
    data = request.get_json()
    image = decode_image(data['image'])
    result = high_knee.analyze_high_knee(image)
    return jsonify(result)




from .detectors import lunges

@bp.route('/analyze/lunges', methods=['POST'])
def analyze_lunges():
    data = request.get_json()
    image = decode_image(data['image'])
    result = lunges.analyze_lunges(image)
    return jsonify(result)

