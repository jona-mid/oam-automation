import pandas as pd
import argparse
import ee
from utils import parse_bbox_string

_worldcover = None
_forest_mask = None


def _get_forest_mask():
    """Get or create forest mask."""
    global _worldcover, _forest_mask
    if _forest_mask is None:
        _worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map")
        _forest_mask = _worldcover.eq(10).Or(_worldcover.eq(95))
    return _forest_mask


def init_earthengine(authenticate_if_needed=False):
    """Initialize Earth Engine. Returns True if successful."""
    try:
        ee.Initialize(project="earth-engine-481316")
        return True
    except Exception as e:
        if authenticate_if_needed:
            try:
                ee.Authenticate()
                ee.Initialize(project="earth-engine-481316")
                return True
            except Exception as e2:
                print(f"Earth Engine initialization failed: {e2}")
                return False
        return False


def calculate_forest_percentage(bbox_coords):
    """Compute forest % for bbox coordinates using Earth Engine."""
    if bbox_coords is None:
        return None

    min_lon, min_lat, max_lon, max_lat = bbox_coords
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    forest_mask = _get_forest_mask()

    try:
        stats = forest_mask.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9
        ).getInfo()

        forest_fraction = stats.get("Map", 0)
        return float(forest_fraction) * 100 if forest_fraction is not None else 0.0
    except Exception as e:
        print(f"Error calculating forest percentage: {e}")
        return None


def filter_openaerial_data(
    df,
    max_gsd_cm=10,
    uploaded_after_date=None,
    uploaded_before_date=None,
    platform_type=["uav", "aircraft"],
    forest_percentage_min=None,
    forest_percentage_max=None,
):
    """
    Filters the OpenAerialMap DataFrame based on specified criteria.

    Args:
        df (pd.DataFrame): The DataFrame containing OpenAerialMap metadata.
        max_gsd_cm (float): Maximum allowed Ground Sample Distance (GSD) in centimeters.
                            Images with GSD smaller than this value are kept.
        uploaded_after_date (str, optional): Filters for images uploaded to OpenAerialMap on or after this date (format 'YYYY-MM-DD').
        uploaded_before_date (str, optional): Filters for images uploaded to OpenAerialMap on or before this date (format 'YYYY-MM-DD').
        platform_type (str or list): Platform type(s) to filter (e.g., 'uav' or ['uav', 'aircraft']). Case-insensitive.
        forest_percentage_min (float, optional): Minimum forest percentage (exclusive, > min). If None, no lower bound.
        forest_percentage_max (float, optional): Maximum forest percentage (inclusive, <= max). If None, no upper bound.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    initial_records = len(df)
    print(f"\nStarting filtering with {initial_records} records.")
    filtered_df = df.copy()

    # 1. Filter by resolution (< max_gsd_cm)
    if "gsd" in filtered_df.columns:
        # Convert to numeric, coercing errors to NaN
        filtered_df["gsd_numeric"] = pd.to_numeric(filtered_df["gsd"], errors="coerce")
        original_count = len(filtered_df)
        filtered_df = filtered_df[
            (filtered_df["gsd_numeric"].notna())
            & (filtered_df["gsd_numeric"] < max_gsd_cm / 100.0)
        ]
        print(
            f"  - After GSD filter (< {max_gsd_cm}cm): {len(filtered_df)} records ({(original_count - len(filtered_df))} removed)."
        )
        filtered_df = filtered_df.drop(
            columns=["gsd_numeric"]
        )  # Clean up temporary column
    else:
        print("  - Warning: 'gsd' column not found for resolution filtering. Skipping.")

    # 2. Filter by property_bands (> 1)
    if "property_bands" in filtered_df.columns:
        original_count = len(filtered_df)
        filtered_df["property_bands_numeric"] = pd.to_numeric(
            filtered_df["property_bands"], errors="coerce"
        )
        filtered_df = filtered_df[
            filtered_df["property_bands_numeric"].notna()
            & (filtered_df["property_bands_numeric"] > 1)
        ]
        print(
            f"  - After property_bands filter (>1): {len(filtered_df)} records ({(original_count - len(filtered_df))} removed)."
        )
        filtered_df = filtered_df.drop(columns=["property_bands_numeric"])
    else:
        print("  - Warning: 'property_bands' column not found. Skipping.")

    # 2. Filter by uploaded_at date (>= uploaded_after_date and <= uploaded_before_date if provided)
    # Assuming 'uploaded_at' exists as a string timestamp.
    if "uploaded_at" in filtered_df.columns and uploaded_after_date is not None:
        original_count = len(filtered_df)
        # Convert 'uploaded_at' to datetime and compare
        target_date = pd.to_datetime(uploaded_after_date, utc=True)
        filtered_df["uploaded_datetime"] = pd.to_datetime(
            filtered_df["uploaded_at"], errors="coerce"
        )

        # Lower bound filter
        date_filter = (filtered_df["uploaded_datetime"].notna()) & (
            filtered_df["uploaded_datetime"] >= target_date
        )

        # Upper bound filter (if provided)
        if uploaded_before_date:
            upper_date = pd.to_datetime(uploaded_before_date, utc=True)
            date_filter = date_filter & (filtered_df["uploaded_datetime"] <= upper_date)
            date_range_str = f">= {uploaded_after_date} and <= {uploaded_before_date}"
        else:
            date_range_str = f">= {uploaded_after_date}"

        filtered_df = filtered_df[date_filter]
        print(
            f"  - After uploaded date filter ({date_range_str}): {len(filtered_df)} records ({(original_count - len(filtered_df))} removed)."
        )
        filtered_df = filtered_df.drop(
            columns=["uploaded_datetime"]
        )  # Drop the temporary column
    elif "uploaded_at" not in filtered_df.columns:
        print(
            "  - Warning: 'uploaded_at' column not found for uploaded date filtering. Skipping."
        )

    # 3. Filter by platform (e.g., 'uav' or ['uav', 'aircraft'])
    # Assuming 'platform' column exists. Perform case-insensitive comparison.
    if "platform" in filtered_df.columns and platform_type:
        original_count = len(filtered_df)

        # Handle both string and list inputs
        if isinstance(platform_type, str):
            platform_types = [platform_type.lower()]
        else:
            platform_types = [
                p.lower() if isinstance(p, str) else str(p).lower()
                for p in platform_type
            ]

        filtered_df = filtered_df[
            (filtered_df["platform"].notna())
            & (filtered_df["platform"].str.lower().isin(platform_types))
        ]
        platform_str = (
            platform_type
            if isinstance(platform_type, str)
            else ", ".join(platform_type)
        )
        print(
            f"  - After platform filter ('{platform_str}'): {len(filtered_df)} records ({(original_count - len(filtered_df))} removed)."
        )
    else:
        print(
            "  - Warning: 'platform' column not found for platform filtering. Skipping."
        )

    # 4. Filter by forest_percentage_gee (if bounds are provided and column exists)
    if "forest_percentage_gee" in filtered_df.columns and (
        forest_percentage_min is not None or forest_percentage_max is not None
    ):
        original_count = len(filtered_df)

        # Build forest percentage filter
        forest_filter = filtered_df["forest_percentage_gee"].notna()

        if forest_percentage_min is not None:
            forest_filter = forest_filter & (
                filtered_df["forest_percentage_gee"] > forest_percentage_min
            )

        if forest_percentage_max is not None:
            forest_filter = forest_filter & (
                filtered_df["forest_percentage_gee"] <= forest_percentage_max
            )

        filtered_df = filtered_df[forest_filter].copy()

        # Build filter description string
        if forest_percentage_min is not None and forest_percentage_max is not None:
            filter_desc = f"> {forest_percentage_min}% and <= {forest_percentage_max}%"
        elif forest_percentage_min is not None:
            filter_desc = f"> {forest_percentage_min}%"
        else:
            filter_desc = f"<= {forest_percentage_max}%"

        print(
            f"  - After forest-based filtering ({filter_desc}): {len(filtered_df)} records ({(original_count - len(filtered_df))} removed)."
        )

    # 5. Filter duplicate bboxes (keep first occurrence)
    if "bbox" in filtered_df.columns:
        original_count = len(filtered_df)
        filtered_df = filtered_df.drop_duplicates(subset=["bbox"], keep="first")
        print(
            f"  - After duplicate bbox filter (keep first): {len(filtered_df)} records ({(original_count - len(filtered_df))} removed)."
        )
    else:
        print("  - Warning: 'bbox' column not found for duplicate filtering. Skipping.")

    print(
        f"\nFiltering complete. {len(filtered_df)} records remaining out of {initial_records} initial records."
    )
    return filtered_df


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Filter OpenAerialMap data based on specified criteria"
    )
    parser.add_argument(
        "--input",
        default="openaerial_data.csv",
        help="Input CSV file (default: openaerial_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="openaerial_data_filtered.csv",
        help="Output CSV file (default: openaerial_data_filtered.csv)",
    )
    parser.add_argument(
        "--max_gsd_cm",
        type=float,
        default=10,
        help="Maximum GSD in centimeters (default: 10)",
    )
    parser.add_argument(
        "--uploaded_after_date",
        type=str,
        default=None,
        help="Filter for images uploaded on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--uploaded_before_date",
        type=str,
        default=None,
        help="Filter for images uploaded on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--platform_type",
        nargs="+",
        default=["uav", "aircraft"],
        help="Platform types to filter (default: uav aircraft)",
    )
    parser.add_argument(
        "--forest_percentage_min",
        type=float,
        default=None,
        help="Minimum forest percentage (exclusive, > min)",
    )
    parser.add_argument(
        "--forest_percentage_max",
        type=float,
        default=None,
        help="Maximum forest percentage (inclusive, <= max)",
    )
    args = parser.parse_args()

    # Define input and output file names
    input_csv_file = args.input
    output_csv_file = args.output

    print(f"Loading data from {input_csv_file}...")
    try:
        df = pd.read_csv(input_csv_file)
        print(f"Successfully loaded {len(df)} records.")
    except FileNotFoundError:
        print(
            f"Error: {input_csv_file} not found. Please run scrape.py first to generate the data."
        )
        return
    except Exception as e:
        print(f"Error loading data from {input_csv_file}: {e}")
        return

    # Apply filters for relevant images
    filtered_df = filter_openaerial_data(
        df,
        max_gsd_cm=args.max_gsd_cm,
        uploaded_after_date=args.uploaded_after_date,
        uploaded_before_date=args.uploaded_before_date,
        platform_type=args.platform_type,
    )

    # Save filtered data
    filtered_df.to_csv(output_csv_file, index=False)
    print(f"Saved filtered dataset to {output_csv_file}")

    # Initialize Earth Engine and calculate forest percentages
    ee_ready = init_earthengine(authenticate_if_needed=False)
    if not ee_ready:
        print(
            "Earth Engine not initialized. Call init_earthengine(authenticate_if_needed=True) or run ee.Authenticate() interactively."
        )
        filtered_df["forest_percentage_gee"] = None
    else:
        # Apply the function to each row of the FILTERED DataFrame to get the forest percentage
        print("\nCalculating forest percentages for filtered records...")
        from tqdm import tqdm

        tqdm.pandas(desc="Calculating forest percentages")
        filtered_df["forest_percentage_gee"] = filtered_df["bbox"].progress_apply(
            lambda x: calculate_forest_percentage(parse_bbox_string(x))
        )

        # Debug: show statistics
        non_null_count = filtered_df["forest_percentage_gee"].notna().sum()
        print(
            f"Calculated forest percentages: {non_null_count} non-null values out of {len(filtered_df)}"
        )
        if non_null_count > 0:
            print(
                f"Forest percentage range: {filtered_df['forest_percentage_gee'].min():.2f}% - {filtered_df['forest_percentage_gee'].max():.2f}%"
            )
            print(
                f"Mean forest percentage: {filtered_df['forest_percentage_gee'].mean():.2f}%"
            )

    # Convert uploaded_at to datetime if it exists (for potential later use)
    if "uploaded_at" in filtered_df.columns:
        filtered_df["uploaded_at"] = pd.to_datetime(
            filtered_df["uploaded_at"], utc=True
        )

    # Apply forest percentage filter if bounds are provided
    if args.forest_percentage_min is not None or args.forest_percentage_max is not None:
        # Check if forest_percentage_gee column exists
        if "forest_percentage_gee" not in filtered_df.columns:
            print(
                "Warning: 'forest_percentage_gee' column not found. Cannot apply forest percentage filter."
            )
            print(
                "Forest percentage calculation may have failed. Check Earth Engine initialization."
            )
        else:
            # Call filter_openaerial_data with only forest parameters to apply forest filter
            # Other filters are skipped because uploaded_after_date=None and we pass the already-filtered data
            filtered_df = filter_openaerial_data(
                filtered_df,
                max_gsd_cm=1000,  # Set high to skip GSD filter
                uploaded_after_date=None,  # Skip date filter
                uploaded_before_date=None,  # Skip date filter
                platform_type=[],  # Empty list to skip platform filter
                forest_percentage_min=args.forest_percentage_min,
                forest_percentage_max=args.forest_percentage_max,
            )

            # Save final filtered results (overwrite openaerial_data_filtered.csv)
            filtered_df.to_csv(output_csv_file, index=False)
            print(f"Processing complete. Final results saved to {output_csv_file}.")
            return

    # Save filtered results (if no forest filtering was applied)
    filtered_df.to_csv(output_csv_file, index=False)
    print(f"Processing complete. Results saved to {output_csv_file}.")


if __name__ == "__main__":
    main()
