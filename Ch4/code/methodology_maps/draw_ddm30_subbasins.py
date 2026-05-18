import geopandas as gpd
import matplotlib.pyplot as plt
import os

def draw_ddm30_subbasins():
    base_dir = r"/Volumes/TU200Pro/Chapter test"
    shp_path = os.path.join(base_dir, "Archived_Old_Files", "basins", "basins_joined.shp")
    out_path = os.path.join(base_dir, "Map_DDM30_Subbasins.png")

    print(f"Loading shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)

    print("Plotting map...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=300)
    ax.set_facecolor('#e0f3f8') # slight ocean color
    
    # Try loading world land borders optionally, but subbasins themselves usually cover land well
    try:
        url = "https://raw.githubusercontent.com/martynafford/natural-earth-geojson/master/110m/physical/ne_110m_land.json"
        world = gpd.read_file(url)
        world.plot(ax=ax, color='white', edgecolor='none', zorder=1)
    except Exception as e:
        print(f"Warning: Could not load world map ({e})")

    # Plot DDM30 Subbasins
    print(f"  Plotting {len(gdf)} DDM30 subbasin polygons...")
    # Plot polygons with border and fill
    gdf.plot(ax=ax, facecolor='#8dd3c7', edgecolor='black', linewidth=0.1, alpha=0.9, zorder=2)
    
    # Custom styling
    ax.set_title("DDM30 Global Subbasins", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=16, fontweight='bold')
    ax.set_ylabel("Latitude", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', color='grey', alpha=0.3, zorder=0)
    
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    plt.tight_layout()

    print(f"Saving bare map to {out_path}...")
    plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done.")

if __name__ == "__main__":
    draw_ddm30_subbasins()
