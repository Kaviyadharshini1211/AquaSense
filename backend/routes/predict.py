import os
import torch
import requests
import numpy as np
from fastapi import APIRouter, HTTPException, Body
from PIL import Image
from torchvision import transforms
from models.model import get_model
import tifffile
import cv2
from sentinelhub import (
    SHConfig, BBox, CRS,
    SentinelHubRequest, MimeType,
    DataCollection, MosaickingOrder,
    SentinelHubCatalog
)
import torch.nn as nn

# Router setup
router = APIRouter()

# SentinelHub Config
config = SHConfig()
config.sh_client_id = '46f9fb38-2686-4ab1-9788-f4492e0c8e76'
config.sh_client_secret = 'd1U9OikpHkfqRXOA6iMRY6pil1wAjml1'
config.sh_client_pool_size = 10
config.sh_client_requests_per_second = 10

# Evalscript for downloading TIFF (6 bands + mask)
evalscript = """//VERSION=3
function setup() {
  return {
    input:  ["B02","B03","B04","B08","B11","B12","dataMask"],
    output: { bands: 7, sampleType: "UINT16" }
  };
}
function evaluatePixel(sample) {
  var valid = sample.dataMask === 1 ? 1 : 0;
  return [
    sample.B02 * 10000 * valid,
    sample.B03 * 10000 * valid,
    sample.B04 * 10000 * valid,
    sample.B08 * 10000 * valid,
    sample.B11 * 10000 * valid,
    sample.B12 * 10000 * valid,
    sample.dataMask * 10000
  ];
}
"""

# Image model
model_path = os.path.join("models", "model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model = get_model(num_classes=3, in_channels=6)
try:
    checkpoint = torch.load(model_path, map_location=device)
    image_model.load_state_dict(checkpoint['model_state_dict'])
    image_model.to(device)
    image_model.eval()
    print("Image model loaded successfully.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading image model: {str(e)}")

# Location model
class LocationClassifier(nn.Module):
    def __init__(self):
        super(LocationClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

location_model = LocationClassifier().to(device)
try:
    location_checkpoint = torch.load("models/model1.pth", map_location=device)
    location_model.load_state_dict(location_checkpoint['model_state_dict'])
    location_model.to(device)
    location_model.eval()
    print("Location model loaded successfully.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading location model: {str(e)}")

# Load scaler and label encoder
scaler = location_checkpoint['scaler']
label_encoder = location_checkpoint['label_encoder']

# Prediction classes
classes = ["Low", "Medium", "High"]

resize = transforms.Resize((224, 224))

def create_valid_bbox(lat, lon, offset=0.15):
    import random
    min_lon = lon - offset + random.uniform(-0.05, 0.05)
    max_lon = lon + offset + random.uniform(-0.05, 0.05)
    min_lat = lat - offset + random.uniform(-0.05, 0.05)
    max_lat = lat + offset + random.uniform(-0.05, 0.05)
    if min_lon >= max_lon: min_lon, max_lon = lon - offset, lon + offset
    if min_lat >= max_lat: min_lat, max_lat = lat - offset, lat + offset
    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

def download_tiff(lat, lon):
    from datetime import datetime, timedelta

    bbox = create_valid_bbox(lat, lon)

    today = datetime.now()
    start_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    catalog = SentinelHubCatalog(config=config)
    hits = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(start_date, end_date),
        limit=1
    )
    if not list(hits):
        raise Exception("No suitable images found in SentinelHub catalog")

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
                mosaicking_order=MosaickingOrder.LEAST_CC,
                maxcc=0.8
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=(256, 256),
        config=config
    )

    tif_data_list = request.get_data()
    tif_data = tif_data_list[0]  # correct: now (256, 256, 7)

    print(f"Downloaded TIFF shape: {tif_data.shape}")

    temp_path = os.path.join("temp_download.tiff")
    tifffile.imwrite(temp_path, tif_data)  # save correctly

    return temp_path

def process_tiff_image(path):
    try:
        img = tifffile.imread(path)
        if img is None or img.ndim != 3 or img.shape[2] < 6:
            raise ValueError(f"Bad shape {None if img is None else img.shape}")
        img = img[:, :, :6]  # use only first 6 bands (skip mask)

        bands = []
        for b in range(6):
            band = img[:, :, b]
            band_resized = cv2.resize(band, (224, 224), interpolation=cv2.INTER_AREA)
            bands.append(band_resized)

        img = np.stack(bands, axis=2).astype('float32')
        img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return torch.zeros(6, 224, 224, dtype=torch.float32)

@router.post("/predict/city")
def predict_from_city(city: dict = Body(...)):
    city_name = city.get("city")
    if not city_name:
        raise HTTPException(status_code=400, detail="City not provided")

    # Geocode city name to coordinates
    geo_url = "https://nominatim.openstreetmap.org/search"
    response = requests.get(geo_url, params={"q": city_name, "format": "json", "limit": 1},
                            headers={"User-Agent": "Mozilla/5.0"})
    if not response.ok or not response.json():
        raise HTTPException(status_code=404, detail="City not found")
    location = response.json()[0]
    lat = float(location["lat"])
    lon = float(location["lon"])
    print(f"Geocoding result: {city_name} -> {lat}, {lon}")

    # Download TIFF
    try:
        tiff_path = download_tiff(lat, lon)
        print(f"Downloaded TIFF file: {tiff_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading TIFF: {str(e)}")

    # Image prediction
    image_tensor = process_tiff_image(tiff_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            outputs = image_model(image_tensor)
            probs_image = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making image prediction: {str(e)}")

    # Location model prediction
    try:
        loc_input = np.array([[lat, lon]])
        loc_input = scaler.transform(loc_input)
        loc_tensor = torch.tensor(loc_input, dtype=torch.float32).to(device)

        with torch.no_grad():
            loc_outputs = location_model(loc_tensor)
            probs_location = torch.softmax(loc_outputs, dim=1)[0].cpu().numpy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making location prediction: {str(e)}")

    # Combine Predictions
    combined_probs = (0.7 * probs_image + 0.3 * probs_location)
    combined_predicted_index = int(np.argmax(combined_probs))
    combined_predicted_class = classes[combined_predicted_index]

    return {
        "city": city_name,
        "latitude": lat,
        "longitude": lon,
        "tiff_file": tiff_path.split("/")[-1],
        "final_prediction": {
            "predicted_class": f"{combined_predicted_class} Depletion",
            "class_probabilities": {
                "Low": round(float(combined_probs[0]), 2),
                "Medium": round(float(combined_probs[1]), 2),
                "High": round(float(combined_probs[2]), 2),
            }
        }
    }
