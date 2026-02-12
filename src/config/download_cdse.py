import requests
import json
import shutil
import os
from getpass import getpass


# Run this script in terminal when doen editing!!!!!


USERNAME = input("enter cdse email:")
PASSWORD = getpass("now password:")

# Search parameters
COLLECTION = "SENTINEL-3"
PRODUCT_TYPE = "SL_2_LST___"
START_DATE = "2020-01-01T00:00:00.000Z"
END_DATE = "2026-01-02T00:00:00.000Z"

bbox_dict = {"ceu":
    [
    -2.94988870276606,
    14.13623145787058,
    22.09729996953098,
    46.67157698577364
  ],
    "midwest":
[
    -104.6398098263424,
    37.20195022149474,
    -104.41178601328565,
    37.33111439729535
  ],
"pampas":
    [
        -67.59125051330516,
        -41.47419963763078,
        -67.26413170602194,
        -41.272852898469026
    ]
}


region = "midwest"

lonmin = bbox_dict[region][0]
latmin = bbox_dict[region][1]
lonmax = bbox_dict[region][2]
latmax = bbox_dict[region][3]

AOI_WKT = f"POLYGON(({lonmin} {latmin}, {lonmax} {latmin}, {lonmax} {latmax}, {lonmin} {latmax}, {lonmin} {latmin}))"

DOWNLOAD_DIR = f"/home/ddkovacs/Downloads/slstr_cdse/{region}"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_token(user, pwd):
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": user,
        "password": pwd,
        "grant_type": "password",
    }
    response = requests.post(token_url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def search_products(token):
    odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    # OData Filter: Collection, Date, AOI, and Product Type
    # We filter by Name contains 'SL_2_LST' to target the specific product type
    filter_query = (
        f"Collection/Name eq '{COLLECTION}' "
        f"and ContentDate/Start gt {START_DATE} "
        f"and ContentDate/Start lt {END_DATE} "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{AOI_WKT}') "
        f"and contains(Name, '{PRODUCT_TYPE}')"
    )

    params = {
        "$filter": filter_query,
        "$orderby": "ContentDate/Start asc",
        "$top": 1000,  # Limit to 5 products for safety
    }

    print(f"Searching for {PRODUCT_TYPE}...")
    response = requests.get(odata_url, params=params)
    response.raise_for_status()
    return response.json().get("value", [])


# --- DOWNLOAD FUNCTION ---
def download_product(token, product_id, product_name):
    # The download URL often requires the zipper service for full products
    download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

    headers = {"Authorization": f"Bearer {token}"}
    filepath = os.path.join(DOWNLOAD_DIR, f"{product_name}.zip")

    if os.path.exists(filepath):
        print(f"Skipping {product_name}, already exists.")
        return

    print(f"Downloading {product_name}...")

    with requests.get(download_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print(f"Finished: {filepath}")


# --- MAIN EXECUTION ---
try:
    print("Authenticating...")
    access_token = get_token(USERNAME, PASSWORD)

    products = search_products(access_token)
    print(f"Found {len(products)} products.")

    for p in products:
        p_id = p['Id']
        p_name = p['Name']
        download_product(access_token, p_id, p_name)

except Exception as e:
    print(f"Error: {e}")