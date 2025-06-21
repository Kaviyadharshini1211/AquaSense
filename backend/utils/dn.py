import os
import pandas as pd
import random
import shutil
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
from rasterio import open as rio_open
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import signal
import sys
from openpyxl import load_workbook
from threading import Lock

from sentinelhub import (
    SHConfig, BBox, CRS,
    SentinelHubRequest, MimeType,
    DataCollection, MosaickingOrder,
    SentinelHubCatalog
)

# ─── Config ─────────────────────────────────────────────────────────
config = SHConfig()
config.sh_client_id     = '46f9fb38-2686-4ab1-9788-f4492e0c8e76'
config.sh_client_secret = 'd1U9OikpHkfqRXOA6iMRY6pil1wAjml1'
config.sh_client_pool_size           = 10
config.sh_client_requests_per_second = 10

# ─── Thresholds & Evalscript ────────────────────────────────────────
LOW_THRESHOLD    = 2.0
MEDIUM_THRESHOLD = 5.0

def classify_by_dtwl(dtwl):
    if dtwl < LOW_THRESHOLD: return 'low'
    if dtwl <= MEDIUM_THRESHOLD: return 'medium'
    return 'high'

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

# ─── Folders ────────────────────────────────────────────────────────
staging_dir    = "dataset"
classified_dir = "classified"
os.makedirs(staging_dir, exist_ok=True)
os.makedirs(classified_dir, exist_ok=True)

# ─── Helpers ────────────────────────────────────────────────────────
def create_valid_bbox(lat, lon, offset=0.15):
    min_lon = lon - offset + random.uniform(-0.05, 0.05)
    max_lon = lon + offset + random.uniform(-0.05, 0.05)
    min_lat = lat - offset + random.uniform(-0.05, 0.05)
    max_lat = lat + offset + random.uniform(-0.05, 0.05)
    if min_lon >= max_lon: min_lon, max_lon = lon - offset, lon + offset
    if min_lat >= max_lat: min_lat, max_lat = lat - offset, lat + offset
    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

def stage_download(lat, lon, date_str):
    base  = datetime.strptime(date_str, "%d-%m-%y")
    start = (base - timedelta(days=30)).strftime("%Y-%m-%d")
    end   = (base + timedelta(days=30)).strftime("%Y-%m-%d")
    bbox  = create_valid_bbox(lat, lon)

    catalog = SentinelHubCatalog(config=config)
    hits = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(start, end),
        limit=1
    )
    if not list(hits):
        return [], None

    req = SentinelHubRequest(
        data_folder=staging_dir,
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(start, end),
            mosaicking_order=MosaickingOrder.LEAST_CC,
            maxcc=0.8
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(256, 256),
        config=config
    )

    try:
        arr = req.get_data()[0]
    except Exception:
        return [], None
    if np.all(arr == 0):
        return [], None

    before = set(os.listdir(staging_dir))
    req.save_data()
    new_folder = (set(os.listdir(staging_dir)) - before).pop()
    path = os.path.join(staging_dir, new_folder)
    tif_files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(('.tif', '.tiff'))
    ]

    valid = []
    for t in tif_files:
        with rio_open(t) as src:
            if not all(
                src.read(i).min() == 0 and src.read(i).max() == 0
                for i in range(1, src.count + 1)
            ):
                valid.append(t)
    return valid, path

def process_row(i, row):
    lat, lon, date, dtwl = row['Latitude'], row['Longitude'], row['Date'], row['DTWL']
    saved, folder = stage_download(lat, lon, date)
    if not saved:
        return None

    src = saved[0]
    name = f"img_{i}_{date.replace('-', '')}.tiff"
    dst  = os.path.join(classified_dir, name)
    try:
        shutil.move(src, dst)
        shutil.rmtree(folder, ignore_errors=True)
        return {
            "Image Name": name,
            "DTWL": dtwl,
            "Depletion Category": classify_by_dtwl(dtwl)
        }
    except Exception:
        return None

# ─── Excel Append Logic ─────────────────────────────────────────────
excel_lock = Lock()

# ─── Excel Append Logic ─────────────────────────────────────────────
def append_to_excel(result):
    excel_file = "classified_groundwater.xlsx"
    if not os.path.exists(excel_file):
        # Create a new file with header if it doesn't exist
        df = pd.DataFrame([result])
        df.to_excel(excel_file, index=False)
    else:
        # Append new data to the existing file
        df = pd.DataFrame([result])
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Directly access the sheet without setting `sheets` manually
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)


# ─── Main ───────────────────────────────────────────────────────────
df = pd.read_excel('groundwater_trimmed1.xlsx')

executor = ThreadPoolExecutor(max_workers=50)
futures = {}

def handle_interrupt(signum, frame):
    print("\n⛔️ KeyboardInterrupt detected. Cancelling futures...")
    for future in futures:
        future.cancel()
    executor.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

try:
    for i, row in df.iterrows():
        future = executor.submit(process_row, i, row)
        futures[future] = i

    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        if future.cancelled():
            continue
        try:
            res = future.result()
            if res:
                append_to_excel(res)  # Append to Excel as each result is processed
        except Exception as e:
            print(f"⚠️ Error during processing: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")

finally:
    executor.shutdown(wait=True)
