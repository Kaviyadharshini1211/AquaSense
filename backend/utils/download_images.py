from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest, MimeType,
    DataCollection, MosaickingOrder
)
import os
from datetime import datetime, timedelta
import random

# ─── Sentinel Hub config ────────────────────────────────────────────────────────
config = SHConfig()
config.sh_client_id     = 'dbdf4953-8c8a-486e-98a3-bf9db54234d7'
config.sh_client_secret = 'MJQ0PLgYYa9JARlxfOxRzRiNlgg6qUGc'

# ─── Regions ─────────────────────────────────────────────────────────────────────
regions = {
    "himachal":    [76.0, 31.0, 78.5, 33.5],
    "uttarakhand": [78.0, 29.0, 80.5, 31.5],
}

# ─── Seasons (mm-dd spans) ───────────────────────────────────────────────────────
seasons = [
    ("12-01", "02-28", "Winter"),
    ("03-01", "05-31", "Spring"),
    ("06-01", "08-31", "Summer"),
    ("09-01", "11-30", "Autumn"),
]

# ─── Parameters ─────────────────────────────────────────────────────────────────
images_per_season = 225
image_size = (512, 512)
years = list(range(2020, datetime.now().year + 1))

# ─── Helper: random date in [start_date,end_date] ───────────────────────────────
def random_date(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    return start + (end - start) * random.random()

# ─── Sentinel‑Hub Evalscript ─────────────────────────────────────────────────────
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02","B03","B04","B08","B11","B12"],
    output: {
      bands: 6,
      sampleType: "UINT16"
    }
  };
}
function evaluatePixel(sample) {
  // Scale the values to the correct range (e.g., multiplying by 10000)
  return [
    sample.B02 * 10000, // Blue
    sample.B03 * 10000, // Green
    sample.B04 * 10000, // Red
    sample.B08 * 10000, // NIR
    sample.B11 * 10000, // SWIR
    sample.B12 * 10000  // SWIR
  ];
}
"""

# ─── Main download loop ──────────────────────────────────────────────────────────
for region, box in regions.items():
    print(f"\n📍 Downloading images for region: {region}")
    region_dir = os.path.join("data", region)
    os.makedirs(region_dir, exist_ok=True)
    downloaded = 0

    for mm_start, mm_end, season_name in seasons:
        for i in range(1, images_per_season + 1):
            # pick a random year
            y = random.choice(years)
            # compute start/end dates for this season‑year
            if season_name == "Winter":
                start_date = f"{y}-12-01"
                end_date   = f"{y+1}-02-28"
            else:
                start_date = f"{y}-{mm_start}"
                end_date   = f"{y}-{mm_end}"

            # generate one random image_date to print and embed
            image_dt = random_date(start_date, end_date)
            image_dt_str = image_dt.strftime("%Y-%m-%d")

            # build filename
            fname = (
                f"{region}_{season_name}_{y}"
                f"_{i:03d}_{image_dt_str}.tiff"
            )
            out_path = os.path.join(region_dir, fname)
            if os.path.exists(out_path):
                print(f"⏩ Skipping {fname} (already exists)")
                continue

            print(
                f"⏳ [{season_name} {y}] img {i}/{images_per_season} ⇒ "
                f"{image_dt_str}  (window {start_date} → {end_date})"
            )

            # jitter bbox
            jitter = 0.3
            jittered = [
                box[0] + random.uniform(-jitter, jitter),
                box[1] + random.uniform(-jitter, jitter),
                box[2] + random.uniform(-jitter, jitter),
                box[3] + random.uniform(-jitter, jitter),
            ]
            bbox = BBox(bbox=jittered, crs=CRS.WGS84)

            try:
                req = SentinelHubRequest(
                    data_folder=region_dir,
                    evalscript=evalscript,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L2A,
                            time_interval=(start_date, end_date),
                            mosaicking_order=MosaickingOrder.LEAST_CC,
                            maxcc=0.1
                        )
                    ],
                    responses=[
                        SentinelHubRequest.output_response("default", MimeType.TIFF)
                    ],
                    bbox=bbox,
                    size=image_size,
                    config=config
                )
                req.save_data()  # downloads to data_folder/default.tiff
                tmp = os.path.join(region_dir, "default.tiff")
                if os.path.exists(tmp):
                    os.rename(tmp, out_path)

                downloaded += 1
                print(f"✅ Saved {fname}")
            except Exception as e:
                print(f"❌ Skipped {fname}: {e}")

    total_needed = images_per_season * len(seasons)
    print(f"🎉 Region {region}: downloaded {downloaded}/{total_needed} images.")
