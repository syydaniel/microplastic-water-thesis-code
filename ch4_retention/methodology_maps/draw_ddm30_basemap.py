import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

def create_ddm30_basemap():
    base_dir = r"/Volumes/TU200Pro/Chapter test"
    csv_path = os.path.join(base_dir, "Archived_Old_Files", "basins", "ddm30_MARINAMulti_September2024.csv")
    out_folder = base_dir
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "Map_DDM30_Basemap.png")

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter out valid lat/lon
    df = df.dropna(subset=['lon', 'lat'])

    print("Creating GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )

    print("Plotting map...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=300)
    ax.set_facecolor('white')
    
    # Try loading world map using direct geojson to avoid Geopandas 1.0 dataset deprecation bug
    try:
        url = "https://raw.githubusercontent.com/martynafford/natural-earth-geojson/master/110m/physical/ne_110m_land.json"
        world = gpd.read_file(url)
        world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5, zorder=1)
        # shade ocean slightly
        ax.set_facecolor('#e0f3f8')
    except Exception as e:
        print(f"Warning: Could not load world map ({e})")

    # Plot DDM30 Subbasins
    print(f"  Plotting {len(gdf)} DDM30 subbasins...")
    gdf.plot(ax=ax, markersize=3, color='#d73027', alpha=0.8, label='DDM30 Subbasins (Outlets)', zorder=2)
    
    # Custom styling
    ax.set_title("DDM30 Global Subbasins Basemap", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=16, fontweight='bold')
    ax.set_ylabel("Latitude", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', color='grey', alpha=0.3, zorder=0)
    
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=14)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    plt.tight_layout()

    print(f"Saving to {out_path}...")
    plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done.")

if __name__ == "__main__":
    create_ddm30_basemap()
