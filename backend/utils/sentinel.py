from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox, DataCollection, bbox_to_dimensions
from datetime import date, timedelta
import os
import uuid


def download_tiff_from_coordinates(lat, lon, size=512):
    # Configure Sentinel Hub
    config = SHConfig()
    config.sh_client_id     = '46f9fb38-2686-4ab1-9788-f4492e0c8e76'
    config.sh_client_secret = 'd1U9OikpHkfqRXOA6iMRY6pil1wAjml1'
    
    if not config.sh_client_id or not config.sh_client_secret:
        raise Exception("SentinelHub credentials not set")

    # Define bounding box with a 0.05° margin around the coordinates
    bbox = BBox(bbox=[lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05], crs=CRS.WGS84)
    dimensions = bbox_to_dimensions(bbox, resolution=10)

    # Generate a unique folder for the output
    folder = os.path.join("downloads", uuid.uuid4().hex)
    os.makedirs(folder, exist_ok=True)

    # Sentinel Hub request to download data
    request = SentinelHubRequest(
        evalscript=""" 
            //VERSION=3
            function setup() {
              return {
                input: ["B02", "B03", "B04", "B08", "B11", "B12"],
                output: { bands: 6 }
              };
            }

            function evaluatePixel(sample) {
              return [sample.B4, sample.B3, sample.B2, sample.B8, sample.B11, sample.B12];
            }
        """,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,  # Use Sentinel-2 Level 2A data
            time_interval=(str(date.today() - timedelta(days=10)), str(date.today())),  # Last 10 days
            mosaicking_order="mostRecent"
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=dimensions,
        config=config,
        data_folder=folder
    )

    # Execute the request and save the data
    try:
        response = request.get_data(save_data=True)
        print(f"Response received: {response}")
    except Exception as e:
        raise Exception(f"Error while downloading Sentinel data: {str(e)}")

    # Now search through the folder for the actual TIFF file
    # Find the deepest folder and locate the response.tiff file
    for root, dirs, files in os.walk(folder):
        if "response.tiff" in files:
            tiff_path = os.path.join(root, "response.tiff")
            print(f"TIFF file found at: {tiff_path}")
            return tiff_path
    
    # If we didn't find the file, raise an error
    raise Exception("TIFF file not found after download.")
