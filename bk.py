import gc
import numpy as np
import pandas as pd
import streamlit as st
from fastai.tabular.all import *

st.set_page_config(page_title="Urban Heat Mitigation AI", layout="wide")
st.title("🏙️ AI-Driven Urban Heat Island Mitigation & Cooling Optimizer")

st.caption(
    "Running on a synthetic urban heat dataset (NASA Earthdata download disabled "
    "for this deployment). Spatial pattern and heat-island physics are simulated "
    "to demonstrate the full mitigation pipeline."
)


@st.cache_resource
def initialize_pipeline():
    """
    Generates a synthetic urban heat island dataset instead of downloading
    from NASA Earthdata. Produces a realistic radial heat-island pattern
    (hotter near a simulated city center, cooler toward the edges) so the
    rest of the app (hotspot selection, mitigation sliders, ML model) works
    identically to the real-data version.
    """
    np.random.seed(42)

    n_points = 4000

    # Simulate a city center around a fixed lat/lon
    center_lat, center_lon = 28.6139, 77.2090  # example city center
    lat_spread, lon_spread = 0.5, 0.5

    latitude = center_lat + np.random.uniform(-lat_spread, lat_spread, n_points)
    longitude = center_lon + np.random.uniform(-lon_spread, lon_spread, n_points)

    dist_from_center_lat = latitude - center_lat
    dist_from_center_lon = longitude - center_lon
    dist_from_center = np.sqrt(dist_from_center_lat**2 + dist_from_center_lon**2)

    # UHI intensity: hottest at center, cooling with distance, plus noise
    max_dist = dist_from_center.max()
    uhi_intensity = (
        6.0 * (1 - dist_from_center / max_dist)
        + np.random.normal(0, 0.6, n_points)
    ).clip(0, None)

    df = pd.DataFrame(
        {
            "longitude": longitude,
            "latitude": latitude,
            "uhi_intensity": uhi_intensity,
        }
    )

    df["dist_from_center_lat"] = df["latitude"] - df["latitude"].mean()
    df["dist_from_center_lon"] = df["longitude"] - df["longitude"].mean()

    # NDVI (vegetation) is lower near the hot center, higher toward edges
    df["ndvi"] = (
        0.1
        + 0.5 * (np.abs(df["dist_from_center_lat"]) + np.abs(df["dist_from_center_lon"]))
        + np.random.normal(0, 0.05, len(df))
    ).clip(-0.1, 0.85)

    # Albedo inversely related to UHI intensity (hot areas = lower reflectivity)
    df["albedo"] = (
        0.15 - 0.05 * df["uhi_intensity"] + np.random.normal(0, 0.02, len(df))
    ).clip(0.08, 0.35)

    # Building density inversely related to vegetation
    df["building_density"] = (
        0.8 - df["ndvi"] + np.random.normal(0, 0.05, len(df))
    ).clip(0.0, 1.0)

    gc.collect()

    try:
        splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(df))
        to_advanced = TabularPandas(
            df,
            procs=[Normalize],
            cont_names=["longitude", "latitude", "ndvi", "albedo", "building_density"],
            y_names="uhi_intensity",
            splits=splits,
        )
        dls_advanced = to_advanced.dataloaders(bs=64)
        learn = tabular_learner(dls_advanced, layers=[256, 128], metrics=rmse)
        learn.fit_one_cycle(4, lr_max=1e-2)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, None

    return df, learn


df, learn_advanced = initialize_pipeline()

if df is not None and learn_advanced is not None:
    st.sidebar.header("📍 Select Target Hotspot")
    hotspots_df = df.sort_values(by="uhi_intensity", ascending=False).head(20).copy()
    hotspot_options = [
        f"Hotspot {i+1}: (Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}) - Baseline: {row['uhi_intensity']:.2f}°C"
        for i, row in hotspots_df.reset_index().iterrows()
    ]

    selected_option = st.sidebar.selectbox("Choose a hotspot to mitigate:", hotspot_options)
    selected_idx = hotspot_options.index(selected_option)
    target_row = hotspots_df.iloc[[selected_idx]].copy()

    st.sidebar.header("🌳 Cooling Interventions")
    tree_canopy = st.sidebar.slider("Increase Tree Canopy (NDVI boost)", 0.0, 0.4, 0.2)
    cool_roofs = st.sidebar.slider("Increase Roof Albedo (Reflectivity)", 0.0, 0.25, 0.15)
    reduce_density = st.sidebar.slider("Reduce Concrete Building Density", 0.0, 0.3, 0.1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Hotspot Profile")
        st.write(f"**Latitude:** {target_row['latitude'].values[0]:.4f}")
        st.write(f"**Longitude:** {target_row['longitude'].values[0]:.4f}")
        st.write(f"**Current NDVI (Vegetation):** {target_row['ndvi'].values[0]:.2f}")
        st.write(f"**Current Surface Albedo:** {target_row['albedo'].values[0]:.2f}")
        st.write(f"**Current Building Density:** {target_row['building_density'].values[0]:.2f}")
        st.metric(label="Baseline UHI Intensity", value=f"{target_row['uhi_intensity'].values[0]:.2f} °C")

    with col2:
        st.subheader("Digital Twin Simulation Results")

        mitigated_row = target_row.copy()
        mitigated_row["ndvi"] = (mitigated_row["ndvi"] + tree_canopy).clip(-0.1, 0.85)
        mitigated_row["albedo"] = (mitigated_row["albedo"] + cool_roofs).clip(0.08, 0.35)
        mitigated_row["building_density"] = (
            mitigated_row["building_density"] - reduce_density
        ).clip(0.0, 1.0)

        try:
            dl_test = learn_advanced.dls.test_dl(mitigated_row)
            predicted_new_uhi, _ = learn_advanced.get_preds(dl=dl_test)

            new_uhi_val = predicted_new_uhi.numpy()[0][0]
            baseline_val = target_row["uhi_intensity"].values[0]
            reduction_val = baseline_val - new_uhi_val

            st.write(f"**Simulated NDVI:** {mitigated_row['ndvi'].values[0]:.2f}")
            st.write(f"**Simulated Albedo:** {mitigated_row['albedo'].values[0]:.2f}")
            st.write(f"**Simulated Building Density:** {mitigated_row['building_density'].values[0]:.2f}")

            st.metric(
                label="New Predicted UHI Intensity",
                value=f"{new_uhi_val:.2f} °C",
                delta=f"-{reduction_val:.2f} °C",
                delta_color="inverse",
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")

    st.subheader("Top Regional Hotspots Breakdown")
    st.dataframe(
        df.sort_values(by="uhi_intensity", ascending=False)
        .head(10)[["latitude", "longitude", "uhi_intensity", "ndvi", "albedo", "building_density"]]
    )
else:
    st.error("Failed to build or read data pipeline. Check the errors above for details.")