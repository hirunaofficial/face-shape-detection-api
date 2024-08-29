# Face Shape Detection API

This is a Flask-based API for detecting face shapes from images using OpenCV and a pre-trained landmark model. The API accepts an image URL, processes the image, and returns the detected face shape.

## Features

- **Face Shape Detection**: Identifies the face shape (Round, Oval, Rectangle, Square, Heart-Shaped, Diamond Shaped) based on facial landmarks.
- **Authorization**: API requires an API key for access.
- **Error Handling**: Handles cases like missing images, failed downloads, and unauthorized access.

## Requirements

- Python 3.x
- Flask
- OpenCV
- NumPy
- Requests

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/hirunaofficial/Face-Shape-Detection-API.git
    cd Face-Shape-Detection-API
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
   
3. **Run the application**:
    ```bash
    python main.py
    ```

5. **Test the API**:
    - Use a tool like Postman or `curl` to send a POST request to `http://127.0.0.1:5000/detect_face_shape` with a JSON payload containing the `image_url`.

## API Usage

### Endpoint

`POST /detect_face_shape`

### Headers

- `Authorization: Bearer <your_api_key>`

### Request Body

```json
{
    "image_url": "https://example.com/path/to/your/image.jpg"
}
```

### Response

#### Success:

```json
{
   "status": "success",
   "shape": "Round Face",
   "cheek_ratio": 0.85,
   "jaw_ratio": 0.75,
   "forehead_ratio": 0.80,
   "chin_ratio": 0.35,
   "head_ratio": 1.20,
   "jaw_angle": 45.0
}

```

#### Error:

```json
{
   "status": "error",
   "status_code": "no_face_detect",
   "message": "No face detected"
}

```

## License

This project is licensed under the GPL-3.0 License. See the LICENSE file for details.


## Contact

- Author: Hiruna Gallage
- Website: [hiruna.dev](https://hiruna.dev)
- Email: [hello@hiruna.dev](mailto:hello@hiruna.dev)