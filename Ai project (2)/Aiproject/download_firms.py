import requests
import pandas as pd
from io import StringIO

# ============================================================
# NASA FIRMS HISTORICAL DATA DOWNLOAD
# ============================================================

# ------------------------------------------------------------
# FIRMS MAP KEY
# ------------------------------------------------------------
# Put your CURRENT FIRMS MAP_KEY here.
#
# IMPORTANT:
# - Do NOT share this key publicly.
# - Do NOT upload it to GitHub.
# - If your old key was exposed, generate a new one.
# ------------------------------------------------------------

MAP_KEY = "7f65f9dd29bbefa337b3a13df746b1dc"


# ------------------------------------------------------------
# FIRMS DATA SOURCE
# ------------------------------------------------------------

SOURCE = "VIIRS_SNPP_NRT"


# ------------------------------------------------------------
# INDIA-REGION BOUNDING BOX
# ------------------------------------------------------------
# west, south, east, north

AREA = "68,6,97,36"


# ------------------------------------------------------------
# TARGET DATE RANGE
# ------------------------------------------------------------
# These MUST match the ERA5 dates.

START_DATE = "2026-09-18"
END_DATE = "2026-09-22"


# ============================================================
# BUILD HISTORICAL FIRMS API URL
# ============================================================

# NASA FIRMS API format:
#
# /api/area/csv/
# [MAP_KEY]/
# [SOURCE]/
# [AREA]/
# [DAY_RANGE]/
# [DATE]
#
# DAY_RANGE = 5
# DATE      = 2026-09-18
#
# Therefore this requests:
#
# 2026-09-18
# 2026-09-19
# 2026-09-20
# 2026-09-21
# 2026-09-22

url = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{MAP_KEY}/{SOURCE}/{AREA}/5/{START_DATE}"
)


# ============================================================
# PRINT REQUEST INFORMATION
# ============================================================

print("=" * 70)
print("NASA FIRMS HISTORICAL DATA DOWNLOAD")
print("=" * 70)

print("Source:", SOURCE)
print("Area:", AREA)
print("Requested period:", START_DATE, "to", END_DATE)

print("\nRequesting historical FIRMS data...")


# ============================================================
# DOWNLOAD DATA
# ============================================================

try:

    response = requests.get(
        url,
        timeout=60
    )

except requests.exceptions.RequestException as e:

    print("\nERROR: Could not connect to NASA FIRMS.")
    print(e)

    raise SystemExit(
        "FIRMS download failed."
    )


print("\nHTTP status:", response.status_code)


# ============================================================
# CHECK RESPONSE
# ============================================================

if response.status_code != 200:

    print("\nNASA FIRMS returned an error:")

    print(response.text)

    raise SystemExit(
        "\nFIRMS download failed."
    )


# ============================================================
# READ CSV
# ============================================================

try:

    df = pd.read_csv(
        StringIO(response.text)
    )

except Exception as e:

    print("\nERROR: Could not read FIRMS CSV data.")

    print(e)

    print("\nServer response:")
    print(response.text[:1000])

    raise SystemExit(
        "CSV reading failed."
    )


# ============================================================
# CHECK WHETHER DATA WAS RETURNED
# ============================================================

print("\nRows received from NASA:", len(df))

if len(df) == 0:

    print("\nWARNING:")
    print("NASA returned zero FIRMS detections.")
    print("The downloaded dataset is empty.")

    raise SystemExit(
        "No FIRMS data returned."
    )


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "acq_date" not in df.columns:

    print("\nERROR:")
    print("The FIRMS response does not contain 'acq_date'.")

    print("\nColumns received:")
    print(df.columns.tolist())

    raise SystemExit(
        "Unexpected FIRMS response."
    )


# ============================================================
# CONVERT DATE
# ============================================================

df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="coerce"
)


# Remove rows with invalid dates

df = df[
    df["acq_date"].notna()
].copy()


# ============================================================
# FILTER EXACT TARGET DATES
# ============================================================

start = pd.Timestamp(
    START_DATE
)

end = pd.Timestamp(
    END_DATE
)

df = df[
    (df["acq_date"] >= start) &
    (df["acq_date"] <= end)
].copy()


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    by=[
        "acq_date",
        "latitude",
        "longitude"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE FILE
# ============================================================

output_file = "firms_india_sep18_22.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# VERIFY RESULT
# ============================================================

print("\n" + "=" * 70)
print("FIRMS DOWNLOAD COMPLETE")
print("=" * 70)

print("\nFinal rows:", len(df))

if len(df) > 0:

    print(
        "First date:",
        df["acq_date"].min().date()
    )

    print(
        "Last date:",
        df["acq_date"].max().date()
    )

    print("\nDetections per day:")

    daily_counts = (
        df["acq_date"]
        .dt.date
        .value_counts()
        .sort_index()
    )

    print(daily_counts)

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    print("\nFirst 5 rows:")

    print(
        df.head()
    )

else:

    print("\nWARNING:")
    print("No FIRMS detections found for the requested dates.")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nSaved as:")
print(output_file)

print("=" * 70)