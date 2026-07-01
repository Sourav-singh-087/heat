import gc
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Urban Heat Mitigation AI",
    layout="wide"
)

st.title("🏙️ AI-Driven Urban Heat Island Mitigation & Cooling Optimizer")

st.caption(
    "Running on a synthetic urban heat dataset "
    "(NASA Earthdata download disabled for deployment). "
    "Spatial pattern and heat-island physics are simulated "
    "to demonstrate the mitigation pipeline."
)


@st.cache_resource
def initialize_pipeline():

    np.random.seed(42)

    n_points = 4000

    center_lat = 28.6139
    center_lon = 77.2090

    lat_spread = 0.5
    lon_spread = 0.5

    latitude = center_lat + np.random.uniform(
        -lat_spread,
        lat_spread,
        n_points,
    )

    longitude = center_lon + np.random.uniform(
        -lon_spread,
        lon_spread,
        n_points,
    )

    dist_lat = latitude - center_lat
    dist_lon = longitude - center_lon

    dist = np.sqrt(dist_lat ** 2 + dist_lon ** 2)

    max_dist = dist.max()

    uhi = (
        6
        * (1 - dist / max_dist)
        + np.random.normal(0, 0.6, n_points)
    ).clip(0)

    df = pd.DataFrame(
        {
            "longitude": longitude,
            "latitude": latitude,
            "uhi_intensity": uhi,
        }
    )

    df["dist_from_center_lat"] = (
        df["latitude"] - df["latitude"].mean()
    )

    df["dist_from_center_lon"] = (
        df["longitude"] - df["longitude"].mean()
    )

    df["ndvi"] = (
        0.1
        + 0.5
        * (
            np.abs(df["dist_from_center_lat"])
            + np.abs(df["dist_from_center_lon"])
        )
        + np.random.normal(0, 0.05, len(df))
    ).clip(-0.1, 0.85)

    df["albedo"] = (
        0.15
        - 0.05 * df["uhi_intensity"]
        + np.random.normal(0, 0.02, len(df))
    ).clip(0.08, 0.35)

    df["building_density"] = (
        0.8
        - df["ndvi"]
        + np.random.normal(0, 0.05, len(df))
    ).clip(0, 1)

    gc.collect()

    features = [
        "longitude",
        "latitude",
        "ndvi",
        "albedo",
        "building_density",
    ]

    X = df[features]

    y = df["uhi_intensity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return df, model


df, model = initialize_pipeline()

if df is not None and model is not None:

    st.sidebar.header("📍 Select Target Hotspot")

    hotspots_df = (
        df.sort_values(
            by="uhi_intensity",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    hotspot_options = [

        f"Hotspot {i+1}: "
        f"(Lat: {row['latitude']:.4f}, "
        f"Lon: {row['longitude']:.4f}) "
        f"- Baseline: {row['uhi_intensity']:.2f}°C"

        for i, row in hotspots_df.reset_index().iterrows()

    ]

    selected_option = st.sidebar.selectbox(
        "Choose a hotspot to mitigate:",
        hotspot_options,
    )

    selected_idx = hotspot_options.index(
        selected_option
    )

    target_row = hotspots_df.iloc[
        [selected_idx]
    ].copy()

    st.sidebar.header("🌳 Cooling Interventions")

    tree_canopy = st.sidebar.slider(
        "Increase Tree Canopy (NDVI boost)",
        0.0,
        0.4,
        0.2,
    )

    cool_roofs = st.sidebar.slider(
        "Increase Roof Albedo",
        0.0,
        0.25,
        0.15,
    )

    reduce_density = st.sidebar.slider(
        "Reduce Concrete Building Density",
        0.0,
        0.3,
        0.1,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Current Hotspot Profile")

        st.write(
            f"**Latitude:** {target_row['latitude'].values[0]:.4f}"
        )

        st.write(
            f"**Longitude:** {target_row['longitude'].values[0]:.4f}"
        )

        st.write(
            f"**Current NDVI (Vegetation):** {target_row['ndvi'].values[0]:.2f}"
        )

        st.write(
            f"**Current Surface Albedo:** {target_row['albedo'].values[0]:.2f}"
        )

        st.write(
            f"**Current Building Density:** {target_row['building_density'].values[0]:.2f}"
        )

        st.metric(
            label="Baseline UHI Intensity",
            value=f"{target_row['uhi_intensity'].values[0]:.2f} °C",
        )

    with col2:

        st.subheader("Digital Twin Simulation Results")

        mitigated_row = target_row.copy()

        mitigated_row["ndvi"] = (
            mitigated_row["ndvi"] + tree_canopy
        ).clip(-0.1, 0.85)

        mitigated_row["albedo"] = (
            mitigated_row["albedo"] + cool_roofs
        ).clip(0.08, 0.35)

        mitigated_row["building_density"] = (
            mitigated_row["building_density"] - reduce_density
        ).clip(0.0, 1.0)

        features = [
            "longitude",
            "latitude",
            "ndvi",
            "albedo",
            "building_density",
        ]

        try:

            new_uhi_val = model.predict(
                mitigated_row[features]
            )[0]

            baseline_val = target_row[
                "uhi_intensity"
            ].values[0]

            reduction_val = (
                baseline_val - new_uhi_val
            )

            st.write(
                f"**Simulated NDVI:** {mitigated_row['ndvi'].values[0]:.2f}"
            )

            st.write(
                f"**Simulated Albedo:** {mitigated_row['albedo'].values[0]:.2f}"
            )

            st.write(
                f"**Simulated Building Density:** {mitigated_row['building_density'].values[0]:.2f}"
            )

            st.metric(
                label="New Predicted UHI Intensity",
                value=f"{new_uhi_val:.2f} °C",
                delta=f"-{reduction_val:.2f} °C",
                delta_color="inverse",
            )

            improvement = (
                reduction_val / baseline_val
            ) * 100

            st.progress(
                min(improvement / 100, 1.0)
            )

            st.write(
                f"Estimated Cooling Improvement: **{improvement:.1f}%**"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

    st.subheader("Top Regional Hotspots Breakdown")

    st.dataframe(

        df.sort_values(
            by="uhi_intensity",
            ascending=False,
        )
        .head(10)[
            [
                "latitude",
                "longitude",
                "uhi_intensity",
                "ndvi",
                "albedo",
                "building_density",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average UHI",
        f"{df['uhi_intensity'].mean():.2f} °C",
    )

    c2.metric(
        "Maximum UHI",
        f"{df['uhi_intensity'].max():.2f} °C",
    )

    c3.metric(
        "Average NDVI",
        f"{df['ndvi'].mean():.2f}",
    )

else:

    st.error(
        "Failed to initialize the Urban Heat Mitigation pipeline."
    )