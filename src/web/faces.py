import os
import sys
import logging
from flask import Flask, jsonify, make_response, request, abort, send_file

# Add web directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import emotion_gender_processor as eg_processor

app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "service": "Face Emotion & Gender Classification API",
        "endpoints": {
            "/classifyImage": "POST an image file with key 'image'"
        }
    })


@app.route('/classifyImage', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return make_response(jsonify({'error': 'No image file uploaded in request'}), 400)
        image_bytes = request.files['image'].read()
        output_file = eg_processor.process_image(image_bytes)
        return send_file(output_file, mimetype='image/png')
    except Exception as err:
        logging.error(f'An error has occurred whilst processing the file: "{err}"')
        abort(400)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'We cannot process the file sent in the request.'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Resource not found.'}), 404)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8084))
    app.run(debug=True, host='0.0.0.0', port=port)
